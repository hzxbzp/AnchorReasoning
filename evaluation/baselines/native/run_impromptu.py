"""Impromptu-VLA-7B (released ``ImpromptuVLA-7B_AD``) native zero-shot on the rated validation frames
-> chains.json rows.

The released checkpoint is a Qwen2.5-VL-7B fine-tuned (LLaMA-Factory, qwen2_vl template) on the
Impromptu + nuScenes planning QA.  Its trajectory task is:
    "You are an autonomous driving agent. You have access to a front view camera image of a vehicle
     <image>. ... predict future waypoints for the vehicle over the next 3 timesteps ... previous ego
     vehicle status recorded over the last 3.0 seconds (at 0.5-second intervals) ... (t-3.0s) [x, y],
     Acceleration: X ax, Y ay m/s^2, Velocity: v m/s, Steering angle: s (...), ..., (t-0.0s) [0.0, 0.0], ..."
    -> "<PLANNING>... [x, y]: p1, ..., p6</PLANNING>"   (6 points @ 0.5 s = 3 s, x forward / y left)
This runner reproduces that prompt from the Waymo frame (front third of the panorama at native
resolution; 3 s of history @ 0.5 s from ``past_states`` including acceleration and speed; the
steering-angle clause is dropped because Waymo carries no steering signal; the annotated intent is
injected as "Driving intent: <go left|go right|go straight>." exactly as every other model receives
it) and records per frame:
    text     = the raw answer (``<PLANNING>...</PLANNING>``)
    pred_xy  = the 6 points @ 0.5 s -> 20 @ 0.25 s (linear through the origin, HELD at the 3 s point
               out to 5 s, because the model's native horizon is shorter than the 5 s asked for)

Run this inside the Impromptu-VLA virtual environment (torch 2.7, transformers 4.52).
The checkpoint location comes from ``ANCHOR_WEIGHTS`` or ``--model-dir``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("native_common", HERE / "common.py")
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)          # type: ignore[union-attr]

MODEL_REL = "ImpromptuVLA-7B_AD"
INTENT_MAP = {"GO_STRAIGHT": "go straight", "GO_LEFT": "go left", "GO_RIGHT": "go right"}
PAIR_RX = re.compile(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]")


def history_clause(frame: dict, n_steps: int = 6):
    """(t-3.0s) ... (t-0.0s) at 0.5 s from the 0.25 s ``past_states`` (relative to the current pose)."""
    ps = frame["past_states"]
    px, py = ps["pos_x"], ps["pos_y"]
    vx, vy = ps.get("vel_x") or [0.0] * len(px), ps.get("vel_y") or [0.0] * len(px)
    ax, ay = ps.get("accel_x") or [0.0] * len(px), ps.get("accel_y") or [0.0] * len(px)
    ox, oy = px[-1], py[-1]
    n = len(px)
    items = []
    for k in range(n_steps, -1, -1):                     # k * 0.5 s ago = 2k samples back
        idx = n - 1 - 2 * k
        if idx < 0:
            continue
        items.append(f"(t-{k * 0.5:.1f}s) [{px[idx] - ox:.2f}, {py[idx] - oy:.2f}], "
                     f"Acceleration: X {ax[idx]:.2f}, Y {ay[idx]:.2f} m/s^2, "
                     f"Velocity: {float(np.hypot(vx[idx], vy[idx])):.2f} m/s")
    span = (len(items) - 1) * 0.5
    return span, ", ".join(items)


def build_prompt(frame: dict) -> str:
    span, hist = history_clause(frame)
    intent = f" Driving intent: {INTENT_MAP.get(frame.get('intent', 'GO_STRAIGHT'), 'go straight')}."
    return ("You are an autonomous driving agent. You have access to a front view camera image of a vehicle <image>. "
            "Your task is to do your best to predict future waypoints for the vehicle over the next 3 timesteps, "
            f"given the vehicle's intent inferred from the images.{intent}"
            f"Provided are the previous ego vehicle status recorded over the last {span:.1f} seconds (at 0.5-second "
            "intervals). This includes the x and y coordinates of the ego vehicle. Positive x means forward direction "
            f"while positive y means leftwards. The data is presented in the format [x, y]:.{hist}\n")


def parse_planning(text: str):
    """Numeric [x, y] pairs of the answer (inside <PLANNING> when present) -> list of (x, y)."""
    m = re.search(r"<PLANNING>(.*?)(</PLANNING>|$)", text, re.S)
    body = m.group(1) if m else text
    pts = [(float(a), float(b)) for a, b in PAIR_RX.findall(body)]
    return pts or None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=None, help=f"checkpoint dir (default ANCHOR_WEIGHTS/{MODEL_REL})")
    ap.add_argument("--index", default=C.VAL456_INDEX, help="JSONL index of the rated validation frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=C.MAX_NEW_TOKENS)
    ap.add_argument("--full-panorama", action="store_true", help="feed the 3-camera panorama instead of the front third")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model_dir = args.model_dir or C.staged_or(os.path.join(C.WEIGHTS, MODEL_REL))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    C.seed_everything(args.seed)
    print(f"[load] {MODEL_REL} <- {model_dir}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=dev).eval()
    proc = AutoProcessor.from_pretrained(model_dir)
    print(f"  loaded, VRAM {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    frames = C.load_val456(index=args.index, subset=args.subset)
    sink = C.RowSink(args.out, resume=not args.no_resume)
    print(f"[run] {len(frames)} frames, image={'panorama' if args.full_panorama else 'front third'}, "
          f"resume={len(sink.rows)} done", flush=True)
    for i, fr in enumerate(frames):
        if sink.has(fr):
            continue
        meta = json.load(open(os.path.join(fr["fdir"], "frame.json")))
        user = build_prompt(meta)
        # LLaMA-Factory (qwen2_vl template) puts the image tokens at the inline <image> position
        prompt = proc.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        prompt = prompt.replace("<image>", "<|vision_start|><|image_pad|><|vision_end|>")
        img = Image.open(os.path.join(fr["fdir"], "panorama_geo.png")).convert("RGB")
        if not args.full_panorama:
            w, h = img.size; t = w // 3
            img = img.crop((t, 0, 2 * t, h))                       # FRONT camera = middle third
        err = None; pred_xy = None; out = ""; n_new = 0; prompt_len = None; pts = None
        try:
            inputs = proc(text=[prompt], images=[img], return_tensors="pt").to(dev)
            prompt_len = int(inputs.input_ids.shape[1])
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            new_ids = gen[0, prompt_len:]
            n_new = int(new_ids.numel())
            out = proc.batch_decode(gen[:, prompt_len:], skip_special_tokens=True)[0]
            pts = parse_planning(out)
            if pts:
                pred_xy = C.resample_to_4hz(pts, 0.5)
        except Exception as e:
            err = repr(e)
            print(f"[err] {fr['scene_id']}-{fr['frame_id']}: {err}", flush=True)
        row = C.make_row(fr, out, pred_xy, n_new, n_new >= args.max_new_tokens, prompt_len,
                         native={"points_0p5s": pts, "n_points": 0 if not pts else len(pts), "horizon_s": 0.0 if not pts else 0.5 * len(pts),
                                 "prompt": user if i < 3 else None, "image": "panorama" if args.full_panorama else "front_third",
                                 "error": err, "sampling": {"do_sample": False, "seed": args.seed}})
        sink.add(row)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i + 1}/{len(frames)}] new_tok={n_new} pts={0 if not pts else len(pts)}", flush=True)
    outp = sink.close()
    print(f"[done] {json.dumps(C.summarize(sink.rows))} -> {outp}", flush=True)


if __name__ == "__main__":
    main()
