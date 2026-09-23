"""Qwen3-VL family adapter (``qwen3vl_8b``; also the base of Cosmos and Alpamayo).

Conventions: patch 16 / merge 2, points normalized to 0-1000 (``NORM1000_DESC``; the codec
clips the encoded value to [0, 999]), no extra vocabulary, plain
``Qwen3VLForConditionalGeneration`` checkpoints.
"""
from __future__ import annotations

from transformers import Qwen3VLForConditionalGeneration

from training.adapters.base import BackboneAdapter, resolve_weights_dir


class Qwen3VLAdapter(BackboneAdapter):
    """Qwen3-VL-8B-Instruct (``WEIGHTS/Qwen3-VL-8B-Instruct``)."""

    name = "qwen3vl_8b"
    family = "qwen3vl"
    patch = 16
    merge = 2
    point_mode = "norm1000"
    model_cls = Qwen3VLForConditionalGeneration
    base_rel = "Qwen3-VL-8B-Instruct"

    def _resolve_base_path(self) -> str:
        return resolve_weights_dir(self.base_rel)
