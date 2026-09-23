"""Alpamayo-R1-10B / Alpamayo-1.5-10B native zero-shot on the rated validation frames -> chains.json rows.

Runs the model's own pipeline (``waymo_test/eval_alpamayo_r1_on_waymo_val.py`` of the released
Alpamayo 1.5 checkout: 3 cameras x 4 temporal frames from the neighbouring panoramas, 1.6 s of ego
history @ 10 Hz, chain-of-causation rollout followed by the diffusion action expert, K = 1,
top_p 0.9 / top_k 5 / temperature 0.1, seed 42) and records per frame:
    text     = the chain of causation (``extra['cot']``), plus ``Plan: <meta_action>`` when the model
               emits one
    pred_xy  = (64, 3) @ 10 Hz -> (20, 2) @ 4 Hz through the pipeline's own ``alpamayo_to_waymo20``
    n_new_tokens / prompt_len from a thin wrapper around ``model.vlm.generate`` (the only place where
               the token ids are visible from outside the native API)

Run this inside the Alpamayo virtual environment (torch 2.8, transformers 4.57).  The Alpamayo config
names the Qwen3-VL processor, which an offline cache may not hold; those lookups are redirected to the
locally cached Cosmos-Reason2-8B repository (same tokenizer plus 4000 trajectory tokens, vocabulary
155697).  No model code is modified.

Locations: ``ANCHOR_WEIGHTS`` or ``--model-dir`` for the checkpoints, ``ALPAMAYO_REPO`` / ``--repo``
(and ``ALPAMAYO_R1_REPO``) for the released source checkouts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("native_common", HERE / "common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)          # type: ignore[union-attr]

REPO_DEFAULT = os.environ.get("ALPAMAYO_REPO", os.path.join(C.ROOT, "alpamayo1.5"))
REPOS = {"alpamayo_r1_10b": "models--nvidia--Alpamayo-R1-10B", "alpamayo_15_10b": "models--nvidia--Alpamayo-1.5-10B"}
COSMOS_8B = "models--nvidia--Cosmos-Reason2-8B"


def _add_repo_to_path(repo: str) -> None:
    """Put the released source checkouts on ``sys.path`` (the 1.5 repo imports from the R1 one)."""
    r1 = os.environ.get("ALPAMAYO_R1_REPO", os.path.join(C.ROOT, "alpamayo"))
    for p in (repo, os.path.join(repo, "src"), os.path.join(r1, "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _redirect_qwen3vl_processor(target: str) -> None:
    """``*.from_pretrained('Qwen/Qwen3-VL-*')`` -> the local Cosmos-Reason2-8B snapshot (offline)."""
    import transformers
    classes = [transformers.AutoProcessor, transformers.AutoTokenizer, transformers.AutoConfig,
               transformers.AutoImageProcessor]
    for nm in ("Qwen3VLConfig", "Qwen3VLMoeConfig", "Qwen3VLTextConfig"):
        c = getattr(transformers, nm, None)
        if c is not None:
            classes.append(c)
    for cls in classes:
        orig = cls.from_pretrained

        def mk(o):
            def redir(name, *a, **k):
                if isinstance(name, str) and "Qwen3-VL" in name:
                    name = target
                return o(name, *a, **k)
            return redir
        cls.from_pretrained = staticmethod(mk(orig))


def _load_pipeline(repo: str):
    path = Path(repo) / "waymo_test" / "eval_alpamayo_r1_on_waymo_val.py"
    spec = importlib.util.spec_from_file_location("alpamayo_waymo_eval", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)     # type: ignore[union-attr]
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(REPOS), required=True)
    ap.add_argument("--model-dir", default=None, help="snapshot dir (default ANCHOR_WEIGHTS/<repo>/snapshots/*; ANCHOR_BASE_PATH overrides)")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="released Alpamayo 1.5 source checkout (default $ALPAMAYO_REPO)")
    ap.add_argument("--index", default=C.VAL456_INDEX, help="JSONL index of the rated validation frames")
    ap.add_argument("--out", required=True, help="output JSON (chains.json schema)")
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    import torch
    _add_repo_to_path(args.repo)
    _redirect_qwen3vl_processor(C.unique_snapshot(os.path.join(C.WEIGHTS, COSMOS_8B)))
    P = _load_pipeline(args.repo)
    from alpamayo1_5 import helper                                           # noqa: E402
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5                   # noqa: E402

    model_dir = args.model_dir or C.staged_or(C.unique_snapshot(os.path.join(C.WEIGHTS, REPOS[args.model])))
    C.seed_everything(args.seed)
    print(f"[load] {args.model} <- {model_dir}", flush=True)
    model = Alpamayo1_5.from_pretrained(model_dir, dtype=torch.bfloat16).to(args.device).eval()
    processor = helper.get_processor(model.tokenizer)
    print(f"  loaded, VRAM {torch.cuda.memory_allocated(args.device) / 2**30:.1f} GB", flush=True)

    # ---- token accounting: wrap the VLM generate call (sequences never leave the native API) ----
    stats = {}
    _orig_generate = model.vlm.generate

    def _generate(*a, **k):
        ids = k.get("input_ids", a[0] if a else None)
        stats["prompt_len"] = int(ids.shape[1]) if ids is not None else None
        out = _orig_generate(*a, **k)
        seq = getattr(out, "sequences", out)
        stats["n_new_tokens"] = int(seq.shape[1] - (stats["prompt_len"] or 0))
        return out
    model.vlm.generate = _generate

    frames = C.load_val456(index=args.index, subset=args.subset)
    sink = C.RowSink(args.out, resume=not args.no_resume)
    print(f"[run] {len(frames)} frames, resume={len(sink.rows)} done", flush=True)
    for i, fr in enumerate(frames):
        if sink.has(fr):
            continue
        meta = json.load(open(os.path.join(fr["clip"], "frame.json")))
        past, fut = meta["past_states"], meta["future_states"]
        n_past = len(past["pos_x"]); n_fut = len(fut["pos_x"])
        # ---- ego history @ 10 Hz exactly as the native pipeline (main loop of the eval script) ----
        waymo_past_t = np.linspace(-(n_past - 1) * P.WAYMO_DT, 0.0, n_past)
        waymo_past_xyz = np.stack([past["pos_x"], past["pos_y"], past.get("pos_z", [0.0] * n_past)], axis=1).astype(np.float32)
        if "pos_z" not in past:
            waymo_past_xyz[:, 2] = 0.0
        hist_t = np.linspace(-(P.NUM_HIST - 1) * P.MODEL_DT, 0.0, P.NUM_HIST)
        ego_history_xyz = P._interp_xy_to_10hz(waymo_past_t, waymo_past_xyz, hist_t)
        fut_t = np.linspace(P.MODEL_DT, P.NUM_FUT * P.MODEL_DT, P.NUM_FUT)
        waymo_fut_xyz = np.stack([fut["pos_x"], fut["pos_y"], fut.get("pos_z", [0.0] * n_fut)], axis=1).astype(np.float32)
        ego_future_xyz_50 = P._interp_xy_to_10hz(np.linspace(P.WAYMO_DT, n_fut * P.WAYMO_DT, n_fut), waymo_fut_xyz, fut_t)
        full_xyz = np.concatenate([ego_history_xyz, ego_future_xyz_50], axis=0)
        full_yaw = P._heading_from_velocity(full_xyz)
        ego_history_rot = P._yaw_to_R(full_yaw[:P.NUM_HIST])
        # ---- 4 panoramas x 3 cameras ----
        images_per_cam, hist_folders = P.build_history_panoramas(Path(C.VAL_ROOT), fr["partition"], fr["scene_id"], int(fr["frame_id"]))
        data = P.make_model_input(images_per_cam, ego_history_xyz, ego_history_rot, P.RESIZE_HW)
        messages = helper.create_message(frames=data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"],
                                         num_frames_per_camera=P.NUM_FRAMES_PER_CAM)
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                               continue_final_message=True, return_dict=True, return_tensors="pt")
        model_inputs = helper.to_device({"tokenized_data": inputs, "ego_history_xyz": data["ego_history_xyz"],
                                         "ego_history_rot": data["ego_history_rot"]}, args.device)
        stats.clear()
        err = None
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs, top_p=args.top_p, top_k=args.top_k, temperature=args.temperature,
                    num_traj_samples=1, max_generation_length=args.max_new_tokens, return_extra=True)
            pred_64 = pred_xyz.float().cpu().numpy()[0, 0, 0]
            pred_20 = P.alpamayo_to_waymo20(pred_64).tolist()
            fields = {k: (str(np.asarray(v).flatten()[0]) if np.asarray(v).size else "") for k, v in extra.items()}
        except Exception as e:                                       # keep going; the row records the failure
            err = repr(e); pred_64 = None; pred_20 = None; fields = {}
            print(f"[err] {fr['scene_id']}-{fr['frame_id']}: {err}", flush=True)
        text = fields.get("cot", "") or ""
        if fields.get("meta_action"):
            text = f"{text}\nPlan: {fields['meta_action']}".strip()
        n_new = stats.get("n_new_tokens", 0) or 0
        row = C.make_row(fr, text, pred_20, n_new, n_new >= args.max_new_tokens, stats.get("prompt_len"),
                         native={"cot": fields.get("cot"), "meta_action": fields.get("meta_action"), "answer": fields.get("answer"),
                                 "pred_xyz_10hz": None if pred_64 is None else np.round(pred_64, 4).tolist(),
                                 "history_frames": [p.name for p in hist_folders], "error": err,
                                 "sampling": {"top_p": args.top_p, "top_k": args.top_k, "temperature": args.temperature, "seed": args.seed}})
        sink.add(row)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(frames)}] new_tok={n_new}", flush=True)
    out = sink.close()
    print(f"[done] {json.dumps(C.summarize(sink.rows))} -> {out}", flush=True)


if __name__ == "__main__":
    main()
