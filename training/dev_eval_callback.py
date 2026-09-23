"""In-training dev evaluation callback.

Every `every_steps` optimizer steps (and at train end) rank 0 generates on the fixed dev frame
list (config key `data.dev_online`) with `evaluation.eval_dev.run_dev_eval`, computes the proxy
composite (S1: `s1_composite`, S2: `s2_composite` from `training.core.metrics.composite` -- the
same functions `eval_dev.compute_composite` uses, so the tracker agrees with `metrics.json`) and
saves the best checkpoint PER CURRICULUM STAGE to `<best_root>/best_stage_<a|b|s2>/`
(S1 stage name read from the train dataset; S2 -> 's2'). The tracker is persisted in
`<best_root>/best.json` so a requeue restart does not reset it. Every eval's metrics are also
appended to `<best_root>/../eval_online/history.jsonl` (+ one json per step).

If `evaluation.eval_dev` is missing the callback prints a warning and becomes a no-op (training
never crashes because of it).

Distributed-safe: only rank 0 generates; all ranks hit a barrier afterwards. Generation under
ZeRO-2 is fine (params replicated); ZeRO-3 would need a parameter gather.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Optional

import torch.distributed as dist
from transformers import TrainerCallback

try:
    from evaluation.eval_dev import run_dev_eval
except Exception as _e:  # pragma: no cover - the evaluation package is optional at import time
    run_dev_eval = None
    _IMPORT_ERR = repr(_e)
else:
    _IMPORT_ERR = ""

from training.core.metrics.composite import s1_composite, s2_composite
from training.core.trainer import write_save_extras


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def load_frames(path: str) -> list[str]:
    """Read the dev frame list JSON: a list of frame dirs, or a mapping with key frames/fdirs/dirs,
    or a list of {'fdir': ...} records. Returns a list of frame-dir strings."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for k in ("frames", "fdirs", "dirs", "frame_dirs"):
            if k in obj:
                obj = obj[k]
                break
        else:
            obj = list(obj.keys())
    out = []
    for it in obj:
        if isinstance(it, str):
            out.append(it)
        elif isinstance(it, dict) and "fdir" in it:
            out.append(it["fdir"])
    return out


def composite_of(metrics: dict, stage: str) -> Optional[float]:
    """Proxy composite for `stage` ('s1'|'s2') from `core.metrics.composite`: S1 -> `s1_composite`,
    S2 -> `s2_composite` (flat or nested metric dict); None when the metrics carry none of the
    composite's terms (the functions never raise on a dict; missing terms are skipped)."""
    fn = s1_composite if stage == "s1" else s2_composite
    v = fn(metrics)
    return None if v is None else float(v)


class DevEvalCallback(TrainerCallback):
    """Periodic dev-set generation eval + best-per-stage checkpointing."""

    def __init__(self, processor, adapter, frames: list[str], stage: str = "s1",
                 every_steps: int = 500, max_new_tokens: int = 1024, best_root: Optional[str] = None,
                 train_dataset=None, bad_ids: Optional[list[int]] = None,
                 save_extras: Optional[dict[str, Any]] = None, subset: Optional[int] = None,
                 eval_at_end: bool = True, long_edge: Optional[int] = None):
        """processor/adapter: as used for training; frames: dev frame dirs; stage: 's1'|'s2'
        (= the eval task); train_dataset: to read the current curriculum stage (S1);
        bad_ids: token ids masked at generation; save_extras: json files copied into best dirs;
        long_edge: image long edge (cfg data.long_edge; None -> eval_dev default = 2916)."""
        self.proc = processor
        self.adapter = adapter
        self.frames = list(frames)[: subset] if subset else list(frames)
        self.stage = stage
        self.every = int(every_steps or 0)
        self.max_new_tokens = int(max_new_tokens)
        self.long_edge = int(long_edge) if long_edge else None
        self.best_root = best_root
        self.ds = train_dataset
        self.bad_ids = list(bad_ids) if bad_ids else None
        self.save_extras = dict(save_extras or {})
        self.eval_at_end = eval_at_end
        self.enabled = run_dev_eval is not None and self.every > 0 and len(self.frames) > 0
        if run_dev_eval is None:
            print(f"[dev-eval] WARNING: evaluation.eval_dev.run_dev_eval not importable ({_IMPORT_ERR}); "
                  "online dev eval DISABLED")
        elif not self.frames:
            print("[dev-eval] WARNING: no dev frames given; online dev eval DISABLED")
        self.best: dict[str, float] = {}
        if best_root:
            try:
                with open(os.path.join(best_root, "best.json")) as f:
                    self.best = json.load(f)
            except Exception:
                pass
        self.hist_dir = os.path.join(os.path.dirname(best_root.rstrip("/")), "eval_online") if best_root else None

    # ---- helpers ----
    def _stage_name(self) -> str:
        if self.stage == "s2":
            return self.stage
        if self.ds is not None and hasattr(self.ds, "current_stage_name"):
            try:
                return str(self.ds.current_stage_name())
            except Exception:
                pass
        return "static"

    def _generate_metrics(self, model) -> tuple[dict, list]:
        kw = {"long_edge": self.long_edge} if self.long_edge else {}
        return run_dev_eval(model, self.proc, self.adapter, self.frames, task=self.stage,
                            max_new_tokens=self.max_new_tokens, bad_ids=self.bad_ids, **kw)

    def _eval_and_save(self, state, model, tag: str) -> None:
        was_training = model.training
        model.eval()
        cfg = getattr(model, "config", None)
        prev_cache = getattr(cfg, "use_cache", None)
        if cfg is not None:
            try:
                cfg.use_cache = True   # KV cache for generation (training sets it False)
            except Exception:
                pass
        t0 = time.time()
        try:
            metrics, _ = self._generate_metrics(model)
        except Exception as e:
            print(f"[dev-eval@{tag}] FAILED: {e}\n{traceback.format_exc()}")
            metrics = None
        finally:
            if cfg is not None and prev_cache is not None:
                try:
                    cfg.use_cache = prev_cache
                except Exception:
                    pass
            if was_training:
                model.train()
        if not isinstance(metrics, dict):
            return
        stage = self._stage_name()
        comp = composite_of(metrics, self.stage)
        flat = {f"eval/{k}": v for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if comp is not None:
            flat["eval/composite"] = comp
        print(f"[dev-eval@{tag} stage={stage} n={len(self.frames)} {time.time()-t0:.0f}s] " +
              " ".join(f"{k.split('/', 1)[-1]}={v:.3f}" for k, v in flat.items()))
        self._log_history(state, tag, stage, comp, metrics)
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({**flat, "eval/stage_name": stage}, step=state.global_step)
        except Exception:
            pass
        if comp is not None and self.best_root and comp > self.best.get(stage, -1.0):
            self.best[stage] = comp
            d = os.path.join(self.best_root, f"best_stage_{stage}")
            os.makedirs(d, exist_ok=True)
            model.save_pretrained(d)
            self.proc.save_pretrained(d)
            write_save_extras(d, self.save_extras)
            with open(os.path.join(d, "dev_metrics.json"), "w") as f:
                json.dump({"step": state.global_step, "stage": stage, "composite": comp, "metrics": metrics}, f, indent=2)
            with open(os.path.join(self.best_root, "best.json"), "w") as f:
                json.dump(self.best, f, indent=2)
            print(f"[dev-eval@{tag}] new BEST for stage {stage}: composite={comp:.4f} -> {d}")

    def _log_history(self, state, tag, stage, comp, metrics) -> None:
        if not self.hist_dir:
            return
        try:
            os.makedirs(self.hist_dir, exist_ok=True)
            rec = {"step": state.global_step, "tag": tag, "stage": stage, "composite": comp,
                   "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, list, dict))}}
            with open(os.path.join(self.hist_dir, f"{tag}_{stage}.json"), "w") as f:
                json.dump(rec, f, indent=2)
            with open(os.path.join(self.hist_dir, "history.jsonl"), "a") as f:
                f.write(json.dumps({"step": state.global_step, "tag": tag, "stage": stage, "composite": comp}) + "\n")
        except Exception as e:
            print(f"[dev-eval] history write failed: {e}")

    def _run_synced(self, state, model, tag: str) -> None:
        if state.is_world_process_zero:
            self._eval_and_save(state, model, tag)
        if _is_dist():
            dist.barrier()

    # ---- TrainerCallback hooks ----
    def on_step_end(self, args, state, control, model=None, **kw):
        if self.enabled and model is not None and state.global_step > 0 and state.global_step % self.every == 0:
            self._run_synced(state, model, f"step{state.global_step}")
        return control

    def on_train_end(self, args, state, control, model=None, **kw):
        # only at a real end of training (not a SIGUSR1 stop, which is followed by a requeue)
        if self.enabled and self.eval_at_end and model is not None and state.global_step >= (state.max_steps or 0):
            self._run_synced(state, model, "final")
        return control
