"""Cosmos-Reason2 adapters (``cosmos_2b``, ``cosmos_8b``).

Both snapshots are plain ``Qwen3VLForConditionalGeneration`` checkpoints (Qwen3-VL
architecture, patch 16), so model loading is the standard HF path. The base tokenizer does
NOT contain Alpamayo's ``<iN>`` / trajectory special tokens and this code base never uses
them, so no vocabulary is added (a pure HF model is returned). ``bad_token_ids`` still lists
those tokens if a checkpoint happens to carry them (empty for the base).
"""
from __future__ import annotations

from typing import Any, List, Optional

from training.adapters.base import resolve_weights_dir
from training.adapters.qwen3vl import Qwen3VLAdapter

COSMOS_REPOS = {
    "cosmos_2b": "models--nvidia--Cosmos-Reason2-2B",
    "cosmos_8b": "models--nvidia--Cosmos-Reason2-8B",
}


def cosmos_snapshot(name: str = "cosmos_8b") -> str:
    """Unique snapshot dir of a Cosmos-Reason2 repo under ``WEIGHTS``."""
    return resolve_weights_dir(COSMOS_REPOS[name], snapshot=True)


class CosmosAdapter(Qwen3VLAdapter):
    """Cosmos-Reason2-{2B,8B} (``WEIGHTS/models--nvidia--Cosmos-Reason2-{2B,8B}/snapshots/<unique>``)."""

    def __init__(self, name: str = "cosmos_8b") -> None:
        if name not in COSMOS_REPOS:
            raise ValueError(f"unknown cosmos adapter {name!r}; expected one of {sorted(COSMOS_REPOS)}")
        self.name = name
        self.base_rel = COSMOS_REPOS[name]
        super().__init__()

    def _resolve_base_path(self) -> str:
        return resolve_weights_dir(self.base_rel, snapshot=True)

    def legacy_tokens(self) -> List[str]:
        from training.adapters.alpamayo import alpamayo_legacy_tokens
        return alpamayo_legacy_tokens()

    def extra_vocab_hook(self, tokenizer: Any, model: Optional[Any]) -> None:
        """No tokens are added for Cosmos; only verify the model can index every tokenizer id."""
        self._sync_embeddings(tokenizer, model)
