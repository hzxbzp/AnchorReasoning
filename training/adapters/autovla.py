"""AutoVLA-3B adapter (``autovla_3b``).

AutoVLA (Zewei-Zhou/AutoVLA) is a Qwen2.5-VL-3B-Instruct whose vocabulary was extended by
2048 action-codebook tokens ``<action_0>..<action_2047>`` (vocab 151665 + 2048 = 153713)
and which is distributed as a PyTorch-Lightning ``.ckpt`` (``AutoVLA_PDMS_89.ckpt``), not an
HF directory. Therefore:

* processor / tokenizer come from the Qwen2.5-VL-3B-Instruct HF snapshot, then the 2048
  ``<action_i>`` names are registered (same ids as in the AutoVLA ckpt);
* ``load_model(None)`` builds Qwen2.5-VL-3B, resizes the embeddings to 153713 rows and loads
  the remapped ckpt state dict (Lightning ``autovla.vlm.`` prefix stripped; transformers
  4.49 layout ``visual.* / model.*`` renamed to ``model.visual.* / model.language_model.*``);
* the action tokens are never emitted by this code base and are masked at generation time
  (``bad_token_ids``).
"""
from __future__ import annotations

import os
import time
import warnings
from typing import Any, Dict, List, Optional

import torch

from training.adapters.base import resolve_weights_dir, unique_snapshot
from training.adapters.qwen25 import Qwen25Adapter
from training.core.paths import ROOT, WEIGHTS

ACTION_CODEBOOK_SIZE = 2048
AUTOVLA_CKPT_NAME = "AutoVLA_PDMS_89.ckpt"
QWEN25_3B_REPO = "models--Qwen--Qwen2.5-VL-3B-Instruct"


def action_tokens(n: int = ACTION_CODEBOOK_SIZE) -> List[str]:
    """``['<action_0>', ..., '<action_{n-1}>']``."""
    return [f"<action_{i}>" for i in range(n)]


def remap_autovla_key(k: str) -> str:
    """Lightning/transformers-4.49 key -> transformers-4.57 Qwen2.5-VL key.

    1. strip the AutoVLA wrapper prefixes ``autovla.`` and ``vlm.``;
    2. ``visual.*`` -> ``model.visual.*``; ``model.*`` (LM) -> ``model.language_model.*``;
       ``lm_head.*`` unchanged.
    """
    for p in ("autovla.", "vlm."):
        if k.startswith(p):
            k = k[len(p):]
    if k.startswith("visual."):
        return "model." + k
    if k.startswith("model.") and not k.startswith(("model.language_model.", "model.visual.")):
        return "model.language_model." + k[len("model."):]
    return k


def find_qwen25_3b_snapshot() -> str:
    """Qwen2.5-VL-3B-Instruct HF snapshot: ``WEIGHTS/<repo>`` if present, else ``ROOT/hf_cache/<repo>``."""
    for root in (WEIGHTS, os.path.join(ROOT, "hf_cache")):
        repo = os.path.join(root, QWEN25_3B_REPO)
        if os.path.isdir(os.path.join(repo, "snapshots")):
            return unique_snapshot(repo)
        if os.path.isfile(os.path.join(repo, "config.json")):
            return repo
    raise FileNotFoundError(f"{QWEN25_3B_REPO} not found under {WEIGHTS} or {ROOT}/hf_cache")


class AutoVLAAdapter(Qwen25Adapter):
    """AutoVLA-3B (``WEIGHTS/models--Zewei-Zhou--AutoVLA/snapshots/<unique>``)."""

    name = "autovla_3b"
    base_rel = "models--Zewei-Zhou--AutoVLA"

    def _resolve_base_path(self) -> str:
        return resolve_weights_dir(self.base_rel, snapshot=True)

    def _resolve_processor_path(self) -> str:
        return find_qwen25_3b_snapshot()

    @property
    def ckpt_file(self) -> str:
        """Path of the Lightning checkpoint inside the snapshot."""
        return os.path.join(self.base_path, AUTOVLA_CKPT_NAME)

    # ---- vocabulary ------------------------------------------------------------------------
    def _extend_tokenizer(self, tokenizer: Any) -> int:
        """Register the 2048 action tokens (regular added tokens -> ids 151665..153712, the ids
        the AutoVLA checkpoint was trained with)."""
        return int(tokenizer.add_tokens(action_tokens(), special_tokens=False))

    def legacy_tokens(self) -> List[str]:
        return action_tokens()

    # ---- model ---------------------------------------------------------------------------
    def build_config(self, path: Optional[str] = None) -> Any:
        """Base: Qwen2.5-VL-3B config with the vocabulary enlarged to 153713 rows."""
        from transformers import AutoConfig
        if path:
            return AutoConfig.from_pretrained(path)
        cfg = AutoConfig.from_pretrained(self.processor_path)
        n = self.expected_vocab_size()
        cfg.vocab_size = n
        if getattr(cfg, "text_config", None) is not None and hasattr(cfg.text_config, "vocab_size"):
            cfg.text_config.vocab_size = n
        return cfg

    def expected_vocab_size(self) -> int:
        """151665 (Qwen2.5-VL tokenizer) + 2048 = 153713."""
        return len(self.processor.tokenizer)

    def load_autovla_state_dict(self) -> Dict[str, torch.Tensor]:
        """torch.load the Lightning ckpt (sequential read, no mmap: mmap over a shared network
        filesystem can hang) and remap keys to the transformers-4.57 layout."""
        t0 = time.time()
        try:
            sd = torch.load(self.ckpt_file, map_location="cpu", weights_only=True)
        except Exception as e:                       # Lightning ckpts may pickle non-tensor objects
            warnings.warn(f"weights_only torch.load failed ({type(e).__name__}); retrying with weights_only=False")
            sd = torch.load(self.ckpt_file, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        out = {remap_autovla_key(k): v for k, v in sd.items()}
        print(f"[autovla] loaded {len(out)} tensors from {os.path.basename(self.ckpt_file)} "
              f"in {time.time() - t0:.0f}s", flush=True)
        return out

    def load_model(self, path: Optional[str] = None, dtype: Any = torch.bfloat16,
                   attn: Optional[str] = "flash_attention_2", device_map: Any = None) -> torch.nn.Module:
        """``path`` given -> plain HF load of a fine-tuned checkpoint (vocab already 153713).
        ``path=None`` -> Qwen2.5-VL-3B base + resize to 153713 + AutoVLA ckpt weights."""
        kw = self._from_pretrained_kwargs(dtype, attn, device_map)
        if path:
            return self.model_cls.from_pretrained(path, **kw)
        model = self.model_cls.from_pretrained(self.processor_path, **kw)
        n_vocab = self.expected_vocab_size()
        if int(model.get_input_embeddings().num_embeddings) != n_vocab:
            # mean_resizing=False: the ckpt overwrites every row below, init values are irrelevant
            model.resize_token_embeddings(n_vocab, mean_resizing=False)
        sd = self.load_autovla_state_dict()
        msd = model.state_dict()
        sd = {k: v for k, v in sd.items() if k in msd and tuple(v.shape) == tuple(msd[k].shape)}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_loaded = len(msd) - len(missing)
        print(f"[autovla] {n_loaded}/{len(msd)} tensors loaded from AutoVLA ckpt "
              f"(missing={len(missing)} unexpected={len(unexpected)})", flush=True)
        if n_loaded < 0.95 * len(msd):
            raise RuntimeError(f"AutoVLA ckpt load suspect: only {n_loaded}/{len(msd)} tensors matched "
                               f"(missing e.g. {missing[:5]}) -> check key remap")
        model.config.vocab_size = n_vocab
        if getattr(model.config, "text_config", None) is not None:
            model.config.text_config.vocab_size = n_vocab
        return model
