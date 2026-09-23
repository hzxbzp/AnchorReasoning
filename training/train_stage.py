#!/usr/bin/env python3
"""Stage-1 / Stage-2 supervised fine-tuning entry point.

Assembles: adapter -> processor -> WaymoDataset -> core.sampler.build_sampling (S1 with a
curriculum: StageAwareWeightedSampler; otherwise static stage-2 sample weights) -> TrainCollator ->
WeightedTrainer (+ CurriculumCallback, DevEvalCallback, SignalSaveStop). Auto-resumes from the
latest `checkpoint-*` in the run dir; saves gracefully on SLURM SIGUSR1 (the sbatch requeues);
writes `DONE` when the full run completed. `prompt_hash.json` (hash = `train_ds.prompt_hash()` =
`prompts.prompt_hash(point_desc)`) and `adapter_name.json` are written to the run dir and into
every save directory (trainer hook). `loss.field_weights` is validated here against
`target_builder.W` but applied by the dataset (no global table is ever mutated); the trainer's
per-field log names come from `train_ds.field_names`.

Usage:
  # review: resolve config + build dataset index, NO model load, NO training (exit 2 if the data
  # files are missing; a missing init_from is only a warning here, an error in a real run)
  python training/train_stage.py --config training/configs/qwen25_7b_s1.yaml --print-config
  # real run (2 GPUs, launched by sbatch):
  torchrun --standalone --nproc_per_node=2 training/train_stage.py --config <yaml> [key=value ...]
  # tiny GPU smoke: --smoke  (3 steps, no deepspeed, eval off, 8 frames)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # repository root (this file lives in <root>/training/)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from training.core import paths as P                          # noqa: E402
from training.core.config import load_config, to_yaml, trainer_kwargs   # noqa: E402
from training.core.sampler import build_sampling              # noqa: E402  (single implementation, no local copy)

DEFAULT_WANDB_PROJECT = "anchor-sft"     # fallback when `wandb.project` is not set in the config


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="configs/<model>_<stage>.yaml")
    ap.add_argument("--print-config", action="store_true",
                    help="resolve config + build the dataset, then exit (no model, no training)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run: 3 steps, no deepspeed, no eval, small subset")
    ap.add_argument("--local_rank", type=int, default=-1)   # accepted from launchers
    args, rest = ap.parse_known_args(argv)
    overrides = [o for o in rest if "=" in o and not o.startswith("--")]
    unknown = [o for o in rest if o not in overrides]
    if unknown:
        ap.error(f"unrecognised arguments: {unknown}")
    return args, overrides


def apply_smoke(cfg: dict) -> dict:
    """Overrides for a fast GPU smoke test (3 steps, single process friendly)."""
    tr = cfg["trainer"]
    tr.update({"max_steps": 3, "num_train_epochs": 1, "deepspeed": None, "report_to": "none",
               "save_steps": 1_000_000, "logging_steps": 1, "gradient_accumulation_steps": 2,
               "dataloader_num_workers": 0})
    cfg["eval"]["every_steps"] = 0
    cfg["data"]["long_edge"] = min(int(cfg["data"].get("long_edge", 2916)), 1456)
    cfg["data"]["max_samples"] = 8
    cfg["run_name"] = cfg["run_name"] + "_smoke"
    tr["output_dir"] = os.path.join(P.RUNS, cfg["run_name"])
    tr["run_name"] = cfg["run_name"]
    return cfg


def is_rank0() -> bool:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0) == 0


# ----------------------------------------------------------------------------- assembly
def check_field_weights(cfg: dict) -> dict:
    """Validate `loss.field_weights` against the field weight table `target_builder.W` (unknown
    field -> ValueError) and return it as {field: float}. Does NOT touch the global table:
    WaymoDataset applies the override to its own per-instance copy (`ds.W`), so two datasets in
    one process cannot leak into each other."""
    fw = cfg["loss"].get("field_weights") or {}
    if not fw:
        return {}
    from training.core.target_builder import W
    unknown = [k for k in fw if k not in W]
    if unknown:
        raise ValueError(f"loss.field_weights has unknown fields {unknown} for target_builder.W; "
                         f"known: {list(W)}")
    return {k: float(v) for k, v in fw.items()}


def build_dataset(cfg: dict, adapter):
    """WaymoDataset for cfg['stage']. `loss.field_weights` is only validated here (the dataset
    applies it)."""
    fw = check_field_weights(cfg)
    if fw and is_rank0():
        print(f"[loss] field weight overrides (applied by the dataset): {fw}")
    from training.core.dataset import WaymoDataset
    return WaymoDataset(cfg, adapter, cfg["stage"])


def print_sampling_summary(res: dict, stage: str) -> None:
    """Print what `core.sampler.build_sampling` chose + the realised (weighted) marginals so the
    S1/S2 balance can be verified at launch."""
    import numpy as np
    if res.get("sampler") is not None:
        print(f"[sampling] {stage}: stage-aware sampler (a = intent x context, b = intent x object); "
              f"n={len(res['sampler'])}")
    else:
        w = np.asarray(res.get("sample_weights") or [], dtype=np.float64)
        if w.size:
            print(f"[sampling] {stage}: static weights min(intent*ctx*obj, cap) * motion * attr -> "
                  f"mean-normalised; n={w.size} min={w.min():.3f} max={w.max():.3f}")
    for group, marg in (res.get("marginals") or {}).items():
        for name, dist in (marg or {}).items():
            if isinstance(dist, dict):
                print(f"[sampling] {group}/{name}: " + " ".join(f"{k}={float(v):.3f}" for k, v in dist.items()))


class OptimizerStateDeviceFix:
    """TrainerCallback: after a DeepSpeed *universal* checkpoint load (configs/deepspeed/zero2_universal.json,
    used to resume with a different GPU count) the per-group Adam `step` is restored as the CPU tensor read
    from `zero/<param>/step.pt` (deepspeed/runtime/base_optimizer.py: ``state[p]['step'] = steps[0]``), but
    torch's fused AdamW -- the Transformers default on CUDA -- needs every state tensor on the parameter's
    device and dies at the first optimizer step with ``Expected all tensors to be on the same device, but got
    state_steps is on cpu``. Move it once before the first step. It runs after ``deepspeed_load_checkpoint``
    and before ``optimizer.step`` (Trainer._inner_training_loop order) and is a no-op for ordinary resumes
    and for the non-fused optimizer, which keeps `step` on the CPU on purpose."""

    def __new__(cls):
        from transformers import TrainerCallback

        class _OptimizerStateDeviceFix(TrainerCallback):
            def on_train_begin(self, a, state, control, optimizer=None, **kw):
                import torch
                opt = optimizer
                while opt is not None and hasattr(opt, "optimizer") and getattr(opt, "optimizer") is not opt:
                    opt = opt.optimizer                     # accelerate wrapper -> DeepSpeedZeroOptimizer -> torch AdamW
                if not isinstance(opt, torch.optim.Optimizer):
                    return control
                on_device = any(g.get("fused") or g.get("capturable") for g in opt.param_groups)
                if not on_device:
                    return control
                moved = 0
                for prm, st in opt.state.items():
                    step = st.get("step")
                    if isinstance(step, torch.Tensor) and isinstance(prm, torch.Tensor) and step.device != prm.device:
                        st["step"] = step.to(prm.device)
                        moved += 1
                if moved and state.is_world_process_zero:
                    print(f"[resume] moved {moved} optimizer 'step' tensor(s) to the parameter device (fused AdamW after a universal checkpoint load)", flush=True)
                return control
        return _OptimizerStateDeviceFix()


class SignalSaveStop:
    """TrainerCallback: on SLURM SIGUSR1 (sent ~15 min before walltime) save + stop gracefully;
    the sbatch then requeues and the run resumes from that checkpoint."""

    def __new__(cls):
        from transformers import TrainerCallback

        class _SignalSaveStop(TrainerCallback):
            def __init__(self):
                self._stop = False
                try:
                    signal.signal(signal.SIGUSR1, lambda *_: setattr(self, "_stop", True))
                except Exception:
                    pass

            def on_step_end(self, a, state, control, **kw):
                if self._stop:
                    if state.is_world_process_zero:
                        print(f"[signal] SIGUSR1 received -> saving at step {state.global_step} and stopping")
                    control.should_save = True
                    control.should_training_stop = True
                return control
        return _SignalSaveStop()


def make_save_extras(cfg: dict, adapter, train_ds=None) -> dict:
    """{'prompt_hash.json': ..., 'adapter_name.json': ...} written into the run dir + every save dir.
    The hash is the dataset's own `train_ds.prompt_hash()` (= `prompts.prompt_hash(point_desc)`), so it
    always describes the prompts actually trained on. `train_ds=None` (callers without a dataset) computes the
    same value from the adapter's point codec."""
    desc = adapter.point_codec.desc
    if train_ds is not None:
        h = train_ds.prompt_hash()
    else:
        from training.core.prompts import prompt_hash
        h = prompt_hash(desc)
    return {
        "prompt_hash.json": {"prompt_hash": h, "point_desc": desc,
                             "stage": cfg["stage"], "run_name": cfg["run_name"]},
        "adapter_name.json": {"adapter": adapter.name, "family": adapter.family,
                              "base_path": getattr(adapter, "base_path", None),
                              "stage": cfg["stage"], "run_name": cfg["run_name"],
                              "init_from": cfg.get("init_from"),
                              "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }


def make_trainer(cfg: dict, model, targs, train_ds, collator, processor, sampling: dict,
                 callbacks: list, save_extras: dict):
    """WeightedTrainer wired to the dataset: `field_names = train_ds.field_names` so the
    `loss_field/<name>` logs are named correctly; sampler / sample weights =
    `core.sampler.build_sampling` result."""
    from training.core.trainer import WeightedTrainer
    return WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, data_collator=collator,
        processing_class=processor, train_sample_weights=sampling["sample_weights"],
        train_sampler=sampling["sampler"], lambda_traj=float(cfg["loss"]["lambda_traj"]),
        field_names=list(train_ds.field_names), save_extras=save_extras, callbacks=callbacks or None)


def latest_checkpoint(output_dir: str):
    """Most recent `checkpoint-<step>` dir in output_dir (by step number), or None."""
    cks = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    cks = [c for c in cks if os.path.isdir(c) and c.rsplit("-", 1)[-1].isdigit()]
    if not cks:
        return None
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[-1]))


def apply_wandb_env(cfg: dict) -> None:
    """Honour the advisory `wandb` config block: whenever the trainer reports to wandb, make sure
    WANDB_PROJECT is set (sbatch/env value wins, then `wandb.project`, then DEFAULT_WANDB_PROJECT --
    never left unset); `wandb.enabled: false` also downgrades trainer.report_to to none. Returns
    nothing; reads/writes os.environ."""
    wb = cfg.get("wandb") or {}
    tr = cfg["trainer"]
    rt = tr.get("report_to")
    uses_wandb = rt in ("wandb", "all") or (isinstance(rt, (list, tuple)) and "wandb" in rt)
    if uses_wandb and wb.get("enabled") is False:
        tr["report_to"] = "none"
        uses_wandb = False
    if uses_wandb:
        os.environ.setdefault("WANDB_PROJECT", str(wb.get("project") or DEFAULT_WANDB_PROJECT))


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args, overrides = parse_args(argv)
    cfg = load_config(args.config, overrides)
    if args.smoke:
        cfg = apply_smoke(cfg)
    apply_wandb_env(cfg)
    stage = cfg["stage"]
    out_dir = cfg["trainer"]["output_dir"]
    if is_rank0():
        print(to_yaml(cfg))

    # ---- early sanity checks (cheap, before any heavy import) ----
    check_field_weights(cfg)          # unknown loss.field_weights key -> fail before loading anything
    init_path = cfg.get("init_from") or None
    if init_path and not os.path.isdir(init_path):
        msg = f"init_from does not exist (S2 expects the S1 best dir): {init_path}"
        if args.print_config:
            print(f"[print-config] WARNING: {msg}")
        else:
            raise FileNotFoundError(msg)
    if stage == "s2" and not init_path and is_rank0():
        print("[model] WARNING: S2 without init_from (expected the S1 best dir) -> training from the base checkpoint")

    from training.adapters import get_adapter
    adapter = get_adapter(cfg["adapter"])
    processor = adapter.load_processor()
    try:
        train_ds = build_dataset(cfg, adapter)
        print(f"[data] stage={stage} train frames: {len(train_ds)}")
        sampling = build_sampling(train_ds, cap=float(cfg["sampling"]["combined_cap"]))
        if is_rank0():
            print_sampling_summary(sampling, stage)
    except (FileNotFoundError, ValueError) as e:
        if not args.print_config:
            raise
        print(f"[print-config] FAILED to build the dataset: {type(e).__name__}: {e}")
        return 2

    if args.print_config:
        st = getattr(train_ds, "stats", None)
        if callable(st):
            try:
                print("[data] stats: " + json.dumps(st(), ensure_ascii=False, default=str))
            except Exception as e:
                print(f"[data] stats unavailable: {e}")
        print("[print-config] OK -- no model loaded, no training started.")
        return 0

    # ---- real training path ----
    import torch
    from training.core.trainer import TrainingArguments, write_save_extras
    from training.core.collator import TrainCollator

    model = adapter.load_model(path=init_path, dtype=torch.bfloat16, attn=cfg["model"]["attn"])
    adapter.extra_vocab_hook(processor.tokenizer, model)
    if cfg["model"]["freeze_vision"]:
        adapter.freeze_vision(model)
        print("[model] vision tower FROZEN")
    if cfg["model"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "config"):
            model.config.use_cache = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] {adapter.name} ({adapter.family}) init={init_path or 'base'} trainable={n_train/1e9:.2f}B / {n_all/1e9:.2f}B")

    lr_mult = None
    vmult = float(cfg["model"].get("visual_lr_mult", 1.0))
    if not cfg["model"]["freeze_vision"] and vmult != 1.0:
        lr_mult = {adapter.visual_param_prefix(): vmult}
    targs = TrainingArguments(lr_multiplier=lr_mult, **trainer_kwargs(cfg))

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    collator = TrainCollator(pad_token_id=pad_id)

    save_extras = make_save_extras(cfg, adapter, train_ds)
    if is_rank0():
        os.makedirs(out_dir, exist_ok=True)
        write_save_extras(out_dir, save_extras)
        with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
            f.write(to_yaml(cfg))

    callbacks = []
    cur = cfg.get("curriculum")
    if stage == "s1" and cur and cur.get("schedule"):
        from training.core.curriculum import CurriculumCallback
        callbacks.append(CurriculumCallback(train_ds, list(cur["schedule"]), list(cur["boundaries"])))
    ev = cfg.get("eval") or {}
    if int(ev.get("every_steps", 0) or 0) > 0 and ev.get("enabled", True):
        from training.dev_eval_callback import DevEvalCallback, load_frames
        dev_path = cfg["data"]["dev_online"]
        frames = load_frames(dev_path) if os.path.isfile(dev_path) else []
        if not frames:
            print(f"[dev-eval] WARNING: {dev_path} missing/empty -> online dev eval disabled")
        else:
            callbacks.append(DevEvalCallback(
                processor, adapter, frames, stage=stage,
                every_steps=int(ev["every_steps"]),
                max_new_tokens=int(ev.get("max_new_tokens", 1024)), best_root=os.path.join(out_dir, "best"),
                train_dataset=train_ds, bad_ids=adapter.bad_token_ids(processor.tokenizer),
                save_extras=save_extras, subset=int(ev.get("subset", 0) or 0) or None,
                long_edge=int(cfg["data"].get("long_edge", 0) or 0) or None))
    callbacks.append(OptimizerStateDeviceFix())   # must precede the first optimizer.step (see class doc)
    callbacks.append(SignalSaveStop())

    trainer = make_trainer(cfg, model, targs, train_ds, collator, processor, sampling, callbacks, save_extras)

    resume = latest_checkpoint(out_dir)
    # RESUME_FROM lets the sbatch hand us a node-local copy of that checkpoint: reading DeepSpeed
    # ZeRO optimizer states (~100 GB) straight off a shared parallel filesystem can crawl at a few
    # MB/s, while copying them to node-local scratch first runs at GB/s.
    staged = os.environ.get("RESUME_FROM", "").strip()
    if staged and os.path.isdir(staged):
        print(f"[resume] using node-local staged checkpoint {staged} (shared-filesystem original: {resume})")
        resume = staged
    if resume:
        print(f"[resume] resuming from {resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    # finalise only when the FULL run completed (a SIGUSR1 stop leaves it to the sbatch requeue)
    if trainer.state.global_step >= trainer.state.max_steps:
        trainer.save_model(os.path.join(out_dir, "final"))
        if trainer.is_world_process_zero():
            with open(os.path.join(out_dir, "DONE"), "w") as f:
                f.write(json.dumps({"step": trainer.state.global_step, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}))
        print("[done] training complete, DONE marker written")
    else:
        print(f"[partial] stopped at step {trainer.state.global_step}/{trainer.state.max_steps} "
              f"(checkpoint saved; sbatch will requeue)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
