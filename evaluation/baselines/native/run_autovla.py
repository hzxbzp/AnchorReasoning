"""AutoVLA (Qwen2.5-VL-3B + action tokenizer) native zero-shot on the rated validation frames ->
chains.json rows.

Uses the model's own pipeline (``infer_waymo_autovla.py`` of the released checkout: 3 views x a
4-frame @ 2 Hz video built from the neighbouring panoramas, velocity / acceleration / driving
instruction taken from the frame annotation, the released ``AutoVLA_PDMS_89.ckpt``, the native
chain-of-thought prompt and near-greedy sampling at temperature 0.01) and records per frame:
    text     = the decoded generation with the ``<action_k>`` tokens stripped (the native rationale)
    pred_xy  = 10 waypoints @ 0.5 s decoded from the action tokens -> 20 @ 0.25 s (linear, through
               the origin)
    n_new_tokens / prompt_len / hit_cap from the token ids; ``max_new_tokens`` is the common 1024 cap
               (the native script instead caps ``max_length`` = 2048 including the prompt)

Run this inside the AutoVLA virtual environment (py3.9, torch 2.4, transformers 4.49).
``model.predict`` is re-expressed here step by step (same calls, same arguments) only so the token
counts are visible; no model code is changed.

Locations: ``ANCHOR_WEIGHTS`` or ``--ckpt`` for the checkpoint, ``AUTOVLA_REPO`` / ``--repo`` for the
released source checkout.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("native_common", HERE / "common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)          # type: ignore[union-attr]

REPO_DEFAULT = os.environ.get("AUTOVLA_REPO", os.path.join(C.ROOT, "AutoVLA"))
WEIGHTS_GLOB = "*AutoVLA*"                # hub cache directory of the downloaded AutoVLA repository
CKPT_NAME = "AutoVLA_PDMS_89.ckpt"
ACTION_TOKEN_RX = re.compile(r"<[A-Za-z]+_\d+>")


def _default_ckpt() -> str:
    """The released checkpoint inside the downloaded AutoVLA snapshot under ``ANCHOR_WEIGHTS``."""
    repos = sorted(d for d in glob.glob(os.path.join(C.WEIGHTS, WEIGHTS_GLOB))
                   if os.path.isdir(os.path.join(d, "snapshots")))
    if len(repos) != 1:
        raise FileNotFoundError(f"expected one AutoVLA repository under {C.WEIGHTS}, found {repos}; pass --ckpt")
    return os.path.join(C.staged_or(C.unique_snapshot(repos[0])), CKPT_NAME)


def _load_pipeline(repo: str):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    spec = importlib.util.spec_from_file_location("infer_waymo_autovla", os.path.join(repo, "infer_waymo_autovla.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)     # type: ignore[union-attr]
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None,
                    help=f"Lightning ckpt (default ANCHOR_WEIGHTS/{WEIGHTS_GLOB}/snapshots/*/{CKPT_NAME})")
    ap.add_argument("--repo", default=REPO_DEFAULT, help="released AutoVLA source checkout (default $AUTOVLA_REPO)")
    ap.add_argument("--index", default=C.VAL456_INDEX, help="JSONL index of the rated validation frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--crop-dir", default=None, help="where the per-view crops are cached (default <out dir>/crops)")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    import torch
    P = _load_pipeline(args.repo)
    ckpt = args.ckpt or _default_ckpt()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C.seed_everything(args.seed)
    print(f"[load] AutoVLA ckpt={ckpt} base={P.QWEN}", flush=True)
    model = P.build_model(dev, ckpt=ckpt)
    print(f"  loaded, VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)
    gen = model.gen_conf                                      # native: max_length 2048, T 0.01, top_k 0, top_p 1.0
    tok = model.processor.tokenizer
    eos_ids = {i for i in (tok.eos_token_id, tok.pad_token_id, tok.convert_tokens_to_ids("<|im_end|>")) if i is not None}

    crop_dir = args.crop_dir or os.path.join(os.path.dirname(args.out) or ".", "crops")
    os.makedirs(crop_dir, exist_ok=True)
    frames = C.load_val456(index=args.index, subset=args.subset)
    sink = C.RowSink(args.out, resume=not args.no_resume)
    print(f"[run] {len(frames)} frames, resume={len(sink.rows)} done", flush=True)

    for i, fr in enumerate(frames):
        if sink.has(fr):
            continue
        meta = json.load(open(os.path.join(fr["clip"], "frame.json")))
        ps = meta["past_states"]
        vel = [ps["vel_x"][-1], ps["vel_y"][-1]]
        acc = [ps.get("accel_x", [0])[-1], ps.get("accel_y", [0])[-1]]
        intent = meta.get("intent", "GO_STRAIGHT")
        cmd = P.INTENT_MAP.get(intent, "go straight")
        images = P.build_temporal_images(fr["scene_dir"], fr["scene_id"], fr["frame_id"], crop_dir)
        feats = {"images": images, "vehicle_velocity": vel, "vehicle_acceleration": acc, "driving_command": cmd,
                 "sensor_data_path": ""}
        err = None; pred_xy = None; raw = ""; n_new = 0; prompt_len = None; traj10 = None
        try:
            # ---- == AutoVLA.predict, spelled out so the token counts are visible ----
            inputs = model.get_prompt(feats)
            model_inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            prompt_len = int(inputs.input_ids.shape[1])
            with torch.no_grad():
                outputs = model.vlm.generate(**model_inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
                                             temperature=gen["temperature"], top_k=gen["top_k"], top_p=gen["top_p"])
            trimmed = outputs[0][prompt_len:].cpu()
            n_new = int(trimmed.numel())
            if n_new and int(trimmed[-1]) in eos_ids:          # native drops the final EOS unconditionally
                trimmed = trimmed[:-1]
            raw = model.processor.decode(trimmed)
            action_tokens = trimmed[trimmed >= model.action_start_id]
            traj = model.action_tokenizer.decode_token_ids_to_trajectory(action_tokens)
            if len(traj):
                traj = np.asarray(traj)[0, 1:]                   # (10, 3): x, y, heading @ 0.5 s
                traj10 = traj[:, :2].tolist()
                pred_xy = C.resample_to_4hz(traj10, 0.5)
        except Exception as e:
            err = repr(e)
            print(f"[err] {fr['scene_id']}-{fr['frame_id']}: {err}", flush=True)
        text = ACTION_TOKEN_RX.sub("", raw).strip()
        row = C.make_row(fr, text, pred_xy, n_new, n_new >= args.max_new_tokens, prompt_len,
                         native={"raw": raw, "traj_10x2_0p5s": traj10, "driving_command": cmd,
                                 "error": err, "sampling": {"do_sample": True, "temperature": gen["temperature"],
                                                            "top_k": gen["top_k"], "top_p": gen["top_p"], "seed": args.seed}})
        sink.add(row)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(frames)}] new_tok={n_new}", flush=True)
    out = sink.close()
    print(f"[done] {json.dumps(C.summarize(sink.rows))} -> {out}", flush=True)


if __name__ == "__main__":
    main()
