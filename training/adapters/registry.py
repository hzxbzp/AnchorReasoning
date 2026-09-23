"""Adapter registry: backbone name -> adapter factory."""
from __future__ import annotations

from typing import Callable, Dict

from training.adapters.alpamayo import AlpamayoAdapter
from training.adapters.autovla import AutoVLAAdapter
from training.adapters.base import BackboneAdapter
from training.adapters.cosmos import CosmosAdapter
from training.adapters.impromptu import ImpromptuAdapter
from training.adapters.qwen25 import Qwen25Adapter
from training.adapters.qwen3vl import Qwen3VLAdapter

REGISTRY: Dict[str, Callable[[], BackboneAdapter]] = {
    "qwen25_7b": Qwen25Adapter,
    "qwen3vl_8b": Qwen3VLAdapter,
    "impromptu_7b": ImpromptuAdapter,
    "autovla_3b": AutoVLAAdapter,
    "alpamayo_r1_10b": lambda: AlpamayoAdapter("alpamayo_r1_10b"),
    "alpamayo_15_10b": lambda: AlpamayoAdapter("alpamayo_15_10b"),
    "cosmos_2b": lambda: CosmosAdapter("cosmos_2b"),
    "cosmos_8b": lambda: CosmosAdapter("cosmos_8b"),
}


def get_adapter(name: str) -> BackboneAdapter:
    """Instantiate the adapter registered under ``name`` (raises KeyError for unknown names)."""
    if name not in REGISTRY:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(REGISTRY)}")
    adapter = REGISTRY[name]()
    if adapter.name != name:
        raise RuntimeError(f"registry mismatch: {name!r} produced adapter named {adapter.name!r}")
    return adapter
