"""Weighted-CE Trainer for this code base.

A thin subclass of the Hugging Face ``Trainer`` that changes the loss level only:

  * two-term normalised loss::

        L = L_text + lambda_traj * L_traj
        L_text = sum_{t not in traj} w_t ce_t / sum_{t not in traj} w_t
        L_traj = sum_{t in traj}     w_t ce_t / sum_{t in traj}     w_t

    `traj_mask` (bool, from the dataset/collator) decides which supervised tokens belong to the
    trajectory term. A batch with no trajectory tokens has L = L_text (Stage-1 / chain-only).
  * logs `loss/text`, `loss/traj` and the UNWEIGHTED mean CE per field `loss_field/<name>`.
  * `model_accepts_loss_kwargs = False` -> classic `loss / grad_accum` path (we return a mean).
  * `TrainingArguments.lr_multiplier`: prefix -> LR multiplier (visual tower x0.1), longest match.
  * `_save` writes `prompt_hash.json` + `adapter_name.json` (the `save_extras` mapping) into every
    directory produced by `save_model` / checkpoints, so eval scripts can verify the prompt.

Pure helper `two_term_weighted_ce` carries the whole loss math and runs on CPU; executing this
module as a script performs a small self-check of it.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from transformers import Trainer
from transformers import TrainingArguments as HFTrainingArguments

try:  # field-id -> name for logging (target_builder is another module; optional at import time)
    from training.core.target_builder import FIELDS as _FIELDS
except Exception:  # pragma: no cover - module may not exist yet
    _FIELDS = None

# keys produced by the dataset/collator that the model forward must NOT receive
_NON_MODEL_KEYS = ("loss_weights", "labels", "field_ids", "traj_mask", "task", "fdir", "scale")


@dataclass
class TrainingArguments(HFTrainingArguments):
    lr_multiplier: Optional[dict] = field(
        default=None,
        metadata={"help": "prefix -> LR multiplier, e.g. {'visual': 0.1} / {'model.visual': 0.1}. "
                          "Longest-prefix match on parameter names."},
    )


@dataclass
class LossParts:
    """Result of `two_term_weighted_ce` (all tensors on the logits device)."""
    loss: torch.Tensor          # scalar, differentiable
    l_text: Optional[torch.Tensor]   # None when the batch has no text tokens
    l_traj: Optional[torch.Tensor]   # None when the batch has no trajectory tokens
    ce: torch.Tensor            # (B*(L-1),) per-token unweighted CE (0 where label==-100)
    labels: torch.Tensor        # (B*(L-1),) shifted labels
    text_sel: torch.Tensor      # (B*(L-1),) bool: supervised & not traj
    traj_sel: torch.Tensor      # (B*(L-1),) bool: supervised & traj
    field_ids: Optional[torch.Tensor]  # (B*(L-1),) shifted field ids or None


def two_term_weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor,
                         traj_mask: Optional[torch.Tensor] = None, lambda_traj: float = 1.0,
                         field_ids: Optional[torch.Tensor] = None, eps: float = 1e-6) -> LossParts:
    """Two-term normalised weighted cross-entropy.

    logits (B, L, V) predict labels[:, 1:] (next-token shift). `weights` (B, L) float
    per-token weights (0 on the prompt), `traj_mask` (B, L) bool marks trajectory tokens (None ->
    no trajectory term). Tokens with label -100 never contribute. Returns `LossParts`; `.loss` is
    `L_text + lambda_traj * L_traj` with each term present only if it has positive weight mass.
    """
    B, L, V = logits.shape
    sl = logits[:, :-1, :].contiguous().reshape(-1, V).float()
    slab = labels[:, 1:].contiguous().reshape(-1)
    sw = weights[:, 1:].contiguous().reshape(-1).float()
    ce = F.cross_entropy(sl, slab, ignore_index=-100, reduction="none")
    sup = slab != -100
    if traj_mask is None:
        tm = torch.zeros_like(sup)
    else:
        tm = traj_mask[:, 1:].contiguous().reshape(-1).bool()
    traj_sel = sup & tm
    text_sel = sup & ~tm
    w_text = (sw * text_sel).sum()
    w_traj = (sw * traj_sel).sum()
    l_text = (ce * sw * text_sel).sum() / w_text.clamp_min(eps) if float(w_text) > 0 else None
    l_traj = (ce * sw * traj_sel).sum() / w_traj.clamp_min(eps) if float(w_traj) > 0 else None
    loss = logits.sum() * 0.0   # keeps the graph alive even if everything is masked
    if l_text is not None:
        loss = loss + l_text
    if l_traj is not None:
        loss = loss + float(lambda_traj) * l_traj
    fid = None
    if field_ids is not None:
        fid = field_ids[:, 1:].contiguous().reshape(-1)
    return LossParts(loss=loss, l_text=l_text, l_traj=l_traj, ce=ce, labels=slab,
                     text_sel=text_sel, traj_sel=traj_sel, field_ids=fid)


class WeightedTrainer(Trainer):
    """HF Trainer with the two-term weighted CE, weighted sampling and grouped LRs."""

    def __init__(self, *args, train_sample_weights=None, train_sampler=None,
                 lambda_traj: float = 1.0, field_names: Optional[list] = None,
                 save_extras: Optional[dict[str, Any]] = None, **kwargs):
        """train_sample_weights: per-index weights -> WeightedRandomSampler (with replacement);
        train_sampler: prebuilt sampler (e.g. StageAwareWeightedSampler), takes priority;
        lambda_traj: weight of the trajectory term; field_names: FIELDS list for logging
        (None -> target_builder.FIELDS when importable); save_extras: {filename.json: obj}
        written into every save directory (prompt_hash.json, adapter_name.json)."""
        super().__init__(*args, **kwargs)
        self._sample_weights = train_sample_weights
        self._train_sampler = train_sampler
        # We return a per-microbatch weighted MEAN from compute_loss. Force the Trainer's
        # classic path (loss / grad_accum) instead of the num_items_in_batch SUM convention.
        self.model_accepts_loss_kwargs = False
        if field_names is None and _FIELDS:
            field_names = list(_FIELDS)
        self._init_loss_state(lambda_traj, field_names, save_extras)

    # ---- loss / logging accumulator state ----
    def _init_loss_state(self, lambda_traj: float = 1.0, field_names: Optional[list] = None,
                         save_extras: Optional[dict] = None) -> None:
        """field_names=None -> per-field logs are keyed `loss_field/id<fid>` (no name lookup)."""
        self._lambda_traj = float(lambda_traj)
        self._field_names = list(field_names) if field_names is not None else None
        self._save_extras = dict(save_extras or {})
        self._fl_sum = defaultdict(float)   # per-field unweighted CE sum (for logging)
        self._fl_cnt = defaultdict(int)
        self._term_sum = {"text": 0.0, "traj": 0.0}   # running sums of L_text / L_traj per microbatch
        self._term_cnt = {"text": 0, "traj": 0}

    # ---- sampling ----
    def _get_train_sampler(self, train_dataset=None):
        if self._train_sampler is not None:
            return self._train_sampler
        if self._sample_weights is not None:
            return WeightedRandomSampler(
                weights=torch.as_tensor(self._sample_weights, dtype=torch.double),
                num_samples=len(self._sample_weights), replacement=True)
        return super()._get_train_sampler(train_dataset)

    # ---- loss ----
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        field_ids = inputs.pop("field_ids", None)
        traj_mask = inputs.pop("traj_mask", None)
        for k in _NON_MODEL_KEYS:     # non-tensor metadata (task/fdir) must not reach forward
            inputs.pop(k, None)
        outputs = model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]   # [B, L, V]
        parts = two_term_weighted_ce(logits, labels, weights, traj_mask, self._lambda_traj, field_ids)
        self._accumulate_logs(parts)
        return (parts.loss, outputs) if return_outputs else parts.loss

    def _accumulate_logs(self, parts: LossParts) -> None:
        with torch.no_grad():
            if parts.l_text is not None:
                self._term_sum["text"] += float(parts.l_text); self._term_cnt["text"] += 1
            if parts.l_traj is not None:
                self._term_sum["traj"] += float(parts.l_traj); self._term_cnt["traj"] += 1
            if parts.field_ids is not None:
                sup = parts.text_sel | parts.traj_sel
                for fid in parts.field_ids[sup].unique().tolist():
                    m = sup & (parts.field_ids == fid)
                    self._fl_sum[int(fid)] += float(parts.ce[m].sum())
                    self._fl_cnt[int(fid)] += int(m.sum())

    def collect_extra_logs(self) -> dict[str, float]:
        """Drain the accumulators -> {'loss/text', 'loss/traj', 'loss_field/<name>', ...}."""
        logs: dict[str, float] = {}
        for term in ("text", "traj"):
            if self._term_cnt[term] > 0:
                logs[f"loss/{term}"] = self._term_sum[term] / self._term_cnt[term]
            self._term_sum[term] = 0.0; self._term_cnt[term] = 0
        if self._fl_cnt:
            for fid, cnt in list(self._fl_cnt.items()):
                if cnt <= 0:
                    continue
                name = (self._field_names[fid] if self._field_names and 0 <= fid < len(self._field_names)
                        else f"id{fid}")
                logs[f"loss_field/{name}"] = self._fl_sum[fid] / cnt
            self._fl_sum.clear(); self._fl_cnt.clear()
        return logs

    def log(self, logs, start_time=None):
        logs.update(self.collect_extra_logs())
        ds = getattr(self, "train_dataset", None)
        st = getattr(ds, "_stage", None)
        if st is not None:
            try:
                logs["train/curriculum_stage"] = int(st.value)
            except Exception:
                pass
        return super().log(logs, start_time)

    # ---- per-group learning rates ----
    def create_optimizer(self):
        mult = getattr(self.args, "lr_multiplier", None)
        if not mult or self.optimizer is not None:
            return super().create_optimizer()
        model = self.model
        decay = set(self.get_decay_parameter_names(model))
        groups: dict[tuple, list] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            pref = max((k for k in mult if name.startswith(k)), key=len, default=None)
            m = float(mult[pref]) if pref is not None else 1.0
            wd = self.args.weight_decay if name in decay else 0.0
            groups.setdefault((m, wd), []).append(p)
        if self.optimizer_cls_and_kwargs is not None:
            opt_cls, opt_kw = self.optimizer_cls_and_kwargs
        else:
            opt_cls, opt_kw = self.get_optimizer_cls_and_kwargs(self.args, model)
        param_groups = [{"params": ps, "lr": self.args.learning_rate * m, "weight_decay": wd}
                        for (m, wd), ps in groups.items()]
        self.optimizer = opt_cls(param_groups, **{k: v for k, v in opt_kw.items() if k != "lr"})
        n = {m: sum(p.numel() for p in ps) for (m, _), ps in groups.items()}
        print(f"[optimizer] lr groups (multiplier -> #params): {n}")
        return self.optimizer

    # ---- save hook: prompt_hash.json / adapter_name.json in every save dir ----
    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict=state_dict)
        write_save_extras(output_dir or self.args.output_dir, self._save_extras)


def write_save_extras(output_dir: str, extras: Optional[dict[str, Any]]) -> None:
    """Write each `{filename: json-able obj}` of `extras` into `output_dir` (created if needed)."""
    if not extras:
        return
    os.makedirs(output_dir, exist_ok=True)
    for fname, obj in extras.items():
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":   # CPU self-check of the loss math
    torch.manual_seed(0)
    B, L, V = 2, 8, 12
    logits = torch.randn(B, L, V, requires_grad=True)
    labels = torch.randint(0, V, (B, L)); labels[:, :2] = -100
    w = torch.ones(B, L); w[:, :2] = 0.0; w[0, 5] = 3.0
    tm = torch.zeros(B, L, dtype=torch.bool); tm[:, 5:] = True
    parts = two_term_weighted_ce(logits, labels, w, tm, 1.0)
    parts.loss.backward()
    print(f"loss={parts.loss.item():.4f} text={parts.l_text.item():.4f} traj={parts.l_traj.item():.4f} "
          f"grad={logits.grad is not None}")
