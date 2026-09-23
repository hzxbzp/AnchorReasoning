#!/usr/bin/env python3
"""TrainCollator: pad text-side tensors, concatenate pre-flattened vision tensors.

Ego history is fed to the model as text, so the batch carries no history tensors; the
trajectory token mask ``traj_mask`` is padded with ``False``. Vision inputs are pre-flattened
per image: ``pixel_values`` is ``(sum_patches, dim)`` and ``image_grid_thw`` is
``(num_images, 3)``; both just concatenate across the batch (no padding). Text is right-padded
to the batch max (rounded up to ``pad_to_multiple_of``).

Batch keys: input_ids, attention_mask, labels, loss_weights, field_ids, traj_mask,
pixel_values, image_grid_thw. The per-sample ``task`` string is deliberately NOT forwarded
(the trainer passes the remaining keys straight into ``model(**inputs)``).
"""
from __future__ import annotations

from typing import Dict, List

import torch


class TrainCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad = int(pad_token_id)
        self.mult = int(pad_to_multiple_of or 0)

    def __call__(self, feats: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        maxlen = max(f["input_ids"].size(0) for f in feats)
        if self.mult:
            maxlen = ((maxlen + self.mult - 1) // self.mult) * self.mult
        ids, lbl, wt, fid, tm, attn = [], [], [], [], [], []
        for f in feats:
            n = f["input_ids"].size(0)
            p = maxlen - n
            ids.append(torch.cat([f["input_ids"], torch.full((p,), self.pad, dtype=torch.long)]))
            lbl.append(torch.cat([f["labels"], torch.full((p,), -100, dtype=torch.long)]))
            wt.append(torch.cat([f["loss_weights"].to(torch.float32), torch.zeros(p)]))
            fid.append(torch.cat([f["field_ids"], torch.full((p,), -1, dtype=torch.long)]))
            traj = f.get("traj_mask")
            if traj is None:  # tolerate samples produced without the mask (treated as text)
                traj = torch.zeros(n, dtype=torch.bool)
            tm.append(torch.cat([traj.to(torch.bool), torch.zeros(p, dtype=torch.bool)]))
            attn.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(p, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(lbl),
            "loss_weights": torch.stack(wt),
            "field_ids": torch.stack(fid),
            "traj_mask": torch.stack(tm),
            "pixel_values": torch.cat([f["pixel_values"] for f in feats], dim=0),
            "image_grid_thw": torch.cat([f["image_grid_thw"] for f in feats], dim=0),
        }
