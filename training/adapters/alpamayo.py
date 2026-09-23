"""Alpamayo adapters (``alpamayo_r1_10b``, ``alpamayo_15_10b``).

Alpamayo-R1-10B / Alpamayo-1.5-10B = Cosmos-Reason2-8B (Qwen3-VL architecture) VLM
(``vlm.*`` tensors, vocabulary 155697 = 151669 Qwen3 tokens + 4000 ``<iN>`` trajectory bins +
28 Alpamayo special tokens) + an action expert / diffusion head (``expert.*``, ``action_*``)
which this code base does not use. The inherited tokens are kept in the vocabulary so the
embedding rows load exactly, are never emitted (history, intent and the trajectory are all
plain text) and are masked at generation time.

* ``load_processor``: Cosmos-Reason2-8B processor + the 4000 ``<iN>`` (regular tokens) +
  the 29 original Alpamayo special tokens in the same order as ``alpamayo_r1.models.base_model``
  (``<|image_pad|>`` already exists -> 28 new) => len 155697, ids identical to the
  checkpoint's ``traj_token_ids``.
* ``load_model(None)``: Cosmos-Reason2-8B config with ``vocab_size=155697`` +
  ``vlm.*`` tensors of the snapshot (prefix stripped) via
  ``Qwen3VLForConditionalGeneration.from_pretrained(None, config=, state_dict=)`` -> a pure
  HF model (no trajectory-token fusion, no expert); vocabulary-sized tensors are row-adapted
  to the config size when they differ.
* ``load_model(path)``: plain HF load of a fine-tuned checkpoint directory.
"""
from __future__ import annotations

import os
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from training.adapters.base import read_json, resolve_weights_dir
from training.adapters.cosmos import cosmos_snapshot
from training.adapters.qwen3vl import Qwen3VLAdapter

ALPAMAYO_REPOS = {
    "alpamayo_r1_10b": "models--nvidia--Alpamayo-R1-10B",
    "alpamayo_15_10b": "models--nvidia--Alpamayo-1.5-10B",
}
TRAJ_VOCAB_SIZE = 4000
# Original Alpamayo special-token keys (alpamayo_r1.models.base_model.SPECIAL_TOKENS_KEYS).
# Order matters: ids are assigned sequentially.
ALPAMAYO_SPECIAL_TOKEN_KEYS = [
    "prompt_start", "prompt_end", "image_start", "image_pre_tkn", "image_end",
    "traj_history_start", "traj_history_pre_tkn", "traj_history_end",
    "cot_start", "cot_end", "meta_action_start", "meta_action_end",
    "traj_future_start", "traj_future_pre_tkn", "traj_future_end",
    "traj_history", "traj_future", "image_pad",
    "vectorized_wm", "vectorized_wm_start", "vectorized_wm_end", "vectorized_wm_pre_tkn",
    "route_start", "route_pad", "route_end",
    "question_start", "question_end", "answer_start", "answer_end",
]
ALPAMAYO_SPECIAL_TOKENS = {k: f"<|{k}|>" for k in ALPAMAYO_SPECIAL_TOKEN_KEYS}
# config.json['traj_token_ids'] key -> special-token key
TRAJ_TOKEN_KEYS = {
    "history": "traj_history", "future": "traj_future",
    "history_start": "traj_history_start", "future_start": "traj_future_start",
    "history_end": "traj_history_end", "future_end": "traj_future_end",
}
VLM_PREFIX = "vlm."


def traj_bin_tokens(n: int = TRAJ_VOCAB_SIZE) -> List[str]:
    """``['<i0>', ..., '<i{n-1}>']``."""
    return [f"<i{v}>" for v in range(n)]


def alpamayo_legacy_tokens() -> List[str]:
    """All Alpamayo-only tokens: 4000 ``<iN>`` bins + the special tokens (minus ``<|image_pad|>``,
    which is a genuine Qwen token and must stay generable as part of the prompt only)."""
    specials = [v for k, v in ALPAMAYO_SPECIAL_TOKENS.items() if k != "image_pad"]
    return traj_bin_tokens() + specials


def extend_alpamayo_tokenizer(tokenizer: Any) -> int:
    """Register ``<i0>..<i3999>`` then the special tokens (idempotent); returns #added."""
    n = int(tokenizer.add_tokens(traj_bin_tokens(), special_tokens=False))
    n += int(tokenizer.add_tokens(list(ALPAMAYO_SPECIAL_TOKENS.values()), special_tokens=True))
    return n


def load_prefixed_state_dict(ckpt_dir: str, prefix: str = VLM_PREFIX) -> Dict[str, torch.Tensor]:
    """Read every ``<prefix>*`` tensor of a sharded safetensors checkpoint and strip the prefix.
    Only the requested tensors are read (safe_open), on CPU."""
    from safetensors import safe_open
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        weight_map = read_json(index_path).get("weight_map", {})
        shard_to_keys: Dict[str, List[str]] = defaultdict(list)
        for key, shard in weight_map.items():
            if key.startswith(prefix):
                shard_to_keys[shard].append(key)
    else:
        single = os.path.join(ckpt_dir, "model.safetensors")
        if not os.path.isfile(single):
            raise FileNotFoundError(f"no safetensors index or model.safetensors in {ckpt_dir}")
        with safe_open(single, framework="pt", device="cpu") as f:
            shard_to_keys = {"model.safetensors": [k for k in f.keys() if k.startswith(prefix)]}
    out: Dict[str, torch.Tensor] = {}
    for shard, keys in shard_to_keys.items():
        with safe_open(os.path.join(ckpt_dir, shard), framework="pt", device="cpu") as f:
            for k in keys:
                out[k[len(prefix):]] = f.get_tensor(k)
    if not out:
        raise ValueError(f"no {prefix}* tensors found in {ckpt_dir}")
    return out


def row_adapt_state_dict(sd: Dict[str, torch.Tensor], target_rows: int,
                         keys: Optional[List[str]] = None) -> int:
    """Adapt vocabulary-sized tensors (embed_tokens / lm_head) to ``target_rows`` rows: keep the
    first ``min`` rows of the checkpoint, fill extra rows with the checkpoint's row mean
    (in place; returns #tensors adapted). No-op when sizes already match."""
    keys = keys or [k for k in sd if k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight")]
    n_adapted = 0
    for k in keys:
        t = sd[k]
        if t.dim() != 2 or t.shape[0] == target_rows:
            continue
        new = t.float().mean(0, keepdim=True).expand(target_rows, t.shape[1]).to(t.dtype).clone()
        n = min(target_rows, t.shape[0])
        new[:n] = t[:n]
        sd[k] = new
        n_adapted += 1
        warnings.warn(f"row-adapted {k}: {tuple(t.shape)} -> {tuple(new.shape)}")
    return n_adapted


class AlpamayoAdapter(Qwen3VLAdapter):
    """Alpamayo-R1-10B / Alpamayo-1.5-10B (``WEIGHTS/models--nvidia--Alpamayo-*/snapshots/<unique>``)."""

    def __init__(self, name: str = "alpamayo_r1_10b") -> None:
        if name not in ALPAMAYO_REPOS:
            raise ValueError(f"unknown alpamayo adapter {name!r}; expected one of {sorted(ALPAMAYO_REPOS)}")
        self.name = name
        self.base_rel = ALPAMAYO_REPOS[name]
        super().__init__()
        self.ckpt_config: Dict[str, Any] = read_json(os.path.join(self.base_path, "config.json"))

    def _resolve_base_path(self) -> str:
        return resolve_weights_dir(self.base_rel, snapshot=True)

    def _resolve_processor_path(self) -> str:
        """Tokenizer/processor of the VLM backbone (Cosmos-Reason2-8B); the Alpamayo snapshot
        ships no tokenizer files."""
        return cosmos_snapshot("cosmos_8b")

    # ---- vocabulary ------------------------------------------------------------------------
    def expected_vocab_size(self) -> int:
        """``vocab_size`` recorded in the Alpamayo config.json (155697)."""
        return int(self.ckpt_config["vocab_size"])

    def expected_traj_token_ids(self) -> Dict[str, int]:
        """``traj_token_ids`` recorded in the Alpamayo config.json (name -> id)."""
        return {k: int(v) for k, v in self.ckpt_config.get("traj_token_ids", {}).items()}

    def _extend_tokenizer(self, tokenizer: Any) -> int:
        n = extend_alpamayo_tokenizer(tokenizer)
        self.check_tokenizer(tokenizer)
        return n

    def check_tokenizer(self, tokenizer: Any) -> None:
        """Raise if the extended tokenizer does not reproduce the checkpoint's vocabulary
        (size and trajectory special-token ids)."""
        n, exp = len(tokenizer), self.expected_vocab_size()
        if n != exp:
            raise ValueError(f"{self.name}: tokenizer size {n} != checkpoint vocab {exp}")
        vocab = tokenizer.get_vocab()
        for cfg_key, tok_key in TRAJ_TOKEN_KEYS.items():
            want = self.expected_traj_token_ids().get(cfg_key)
            got = vocab.get(ALPAMAYO_SPECIAL_TOKENS[tok_key])
            if want is not None and got != want:
                raise ValueError(f"{self.name}: id of {ALPAMAYO_SPECIAL_TOKENS[tok_key]} is {got}, "
                                 f"checkpoint expects {want}")
        if vocab.get("<i0>") != int(self.ckpt_config.get("traj_token_start_idx", vocab.get("<i0>"))):
            raise ValueError(f"{self.name}: <i0> id {vocab.get('<i0>')} != traj_token_start_idx "
                             f"{self.ckpt_config.get('traj_token_start_idx')}")

    def legacy_tokens(self) -> List[str]:
        return alpamayo_legacy_tokens()

    def extra_vocab_hook(self, tokenizer: Any, model: Optional[torch.nn.Module]) -> None:
        """Confirm the ``<iN>`` / special tokens are in the vocabulary (add if a foreign tokenizer
        was passed) and that the model has at least as many embedding rows."""
        extend_alpamayo_tokenizer(tokenizer)
        self.check_tokenizer(tokenizer)
        self._sync_embeddings(tokenizer, model)

    # ---- model ---------------------------------------------------------------------------
    def build_config(self, path: Optional[str] = None) -> Any:
        """Base: Cosmos-Reason2-8B (Qwen3-VL) config with ``vocab_size`` = 155697."""
        from transformers import AutoConfig
        if path:
            return AutoConfig.from_pretrained(path)
        cfg = AutoConfig.from_pretrained(self.processor_path)
        n = self.expected_vocab_size()
        cfg.text_config.vocab_size = n
        if hasattr(cfg, "vocab_size"):
            cfg.vocab_size = n
        return cfg

    def load_model(self, path: Optional[str] = None, dtype: Any = torch.bfloat16,
                   attn: Optional[str] = "flash_attention_2", device_map: Any = None) -> torch.nn.Module:
        """``path`` given -> plain HF load of a fine-tuned checkpoint. ``path=None`` -> Qwen3-VL
        architecture from the Cosmos-Reason2-8B config (vocab 155697) + the snapshot's ``vlm.*``
        weights; returns a pure ``Qwen3VLForConditionalGeneration``."""
        kw = self._from_pretrained_kwargs(dtype, attn, device_map)
        if path:
            return self.model_cls.from_pretrained(path, **kw)
        cfg = self.build_config()
        t0 = time.time()
        sd = load_prefixed_state_dict(self.base_path, VLM_PREFIX)
        row_adapt_state_dict(sd, cfg.text_config.vocab_size)
        print(f"[{self.name}] read {len(sd)} vlm.* tensors in {time.time() - t0:.0f}s", flush=True)
        model = self.model_cls.from_pretrained(None, config=cfg, state_dict=sd, **kw)
        missing = [k for k in model.state_dict() if k not in sd]
        if missing:
            warnings.warn(f"{self.name}: {len(missing)} tensors not in checkpoint (fresh init), e.g. {missing[:5]}")
        try:                                    # generation defaults of the base VLM (eos/pad ids)
            from transformers import GenerationConfig
            model.generation_config = GenerationConfig.from_pretrained(self.processor_path)
        except Exception as e:                  # noqa: BLE001 - optional nicety
            warnings.warn(f"{self.name}: could not load generation_config from {self.processor_path}: {e}")
        return model
