"""Impromptu-VLA-7B adapter (``impromptu_7b``).

The checkpoint is a fine-tuned Qwen2.5-VL-7B with an unchanged tokenizer (its
``added_tokens.json`` only lists the standard Qwen2.5-VL specials), so it behaves exactly
like ``qwen25_7b`` apart from the weights directory.
"""
from __future__ import annotations

from training.adapters.qwen25 import Qwen25Adapter


class ImpromptuAdapter(Qwen25Adapter):
    """ImpromptuVLA-7B_AD (``WEIGHTS/ImpromptuVLA-7B_AD``)."""

    name = "impromptu_7b"
    base_rel = "ImpromptuVLA-7B_AD"
