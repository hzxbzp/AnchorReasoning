"""Qwen2.5-VL family adapter (``qwen25_7b``; also the base of Impromptu and AutoVLA).

Conventions: patch 14 / merge 2, points as absolute processed-image pixels, no extra
vocabulary, plain ``Qwen2_5_VLForConditionalGeneration`` checkpoints.
"""
from __future__ import annotations

from transformers import Qwen2_5_VLForConditionalGeneration

from training.adapters.base import BackboneAdapter, resolve_weights_dir


class Qwen25Adapter(BackboneAdapter):
    """Qwen2.5-VL-7B-Instruct (``WEIGHTS/Qwen2.5-VL-7B-Instruct``)."""

    name = "qwen25_7b"
    family = "qwen25"
    patch = 14
    merge = 2
    point_mode = "abs_pixel"
    model_cls = Qwen2_5_VLForConditionalGeneration
    base_rel = "Qwen2.5-VL-7B-Instruct"

    def _resolve_base_path(self) -> str:
        return resolve_weights_dir(self.base_rel)
