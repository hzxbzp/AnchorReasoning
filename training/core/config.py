"""YAML config loader for the supervised fine-tuning pipeline.

Reads a ``configs/*.yaml``, fills every missing key with the documented default (stage-aware:
Stage-2 defaults to lr 5e-6, a frozen vision tower and no curriculum), resolves the path tokens
``PKG`` ``ROOT`` ``RUNS`` ``DATA`` ``WEIGHTS`` ``CACHE`` ``EVAL`` ``LOGS`` (either ``${PKG}/x`` or a
leading ``PKG/x``) against :mod:`training.core.paths`, applies ``key.sub=value`` dot-list overrides
and validates the result. Deliberately imports no torch/transformers so it is cheap to call from
the sbatch shell (``python -c "from training.core.config import load_config; ..."``).

``ROOT`` is the repository root and ``PKG`` is ``<root>/training``; the remaining tokens are the
directories exported by :mod:`training.core.paths` (all environment-overridable).

Public API
----------
load_config(path, overrides=None) -> dict          fully resolved config (plain dict)
default_config(stage) -> dict                      defaults only (no file)
deep_merge(base, override) -> dict                 recursive dict merge (override wins; None overrides)
parse_dotlist(items) -> dict                       ["a.b=1", "c=x"] -> {"a": {"b": 1}, "c": "x"}
resolve_path_tokens(s) -> str                      "PKG/configs/x.yaml" -> "<root>/training/configs/x.yaml"
validate_config(cfg) -> None                       raises ValueError on a broken config
trainer_kwargs(cfg) -> dict                        the `trainer` block ready for TrainingArguments(**)
to_yaml(cfg) -> str
"""
from __future__ import annotations

import copy
import os
import re
from typing import Any, Iterable, Optional

import yaml

from training.core import paths as P

# ----------------------------------------------------------------------------- constants
STAGES = ("s1", "s2")
ADAPTERS = ("qwen25_7b", "qwen3vl_8b", "impromptu_7b", "autovla_3b",
            "alpamayo_r1_10b", "alpamayo_15_10b", "cosmos_2b", "cosmos_8b")
_PATH_TOKENS = {
    "PKG": P.PKG, "ROOT": P.ROOT, "RUNS": P.RUNS, "DATA": P.DATA, "WEIGHTS": P.WEIGHTS,
    "CACHE": P.CACHE, "EVAL": P.EVAL, "LOGS": P.LOGS,
}
# keys whose string values are paths (resolved for tokens)
_PATH_KEYS = {
    ("init_from",),
    ("data", "index"), ("data", "meta"), ("data", "ctx_meta"), ("data", "motion_meta"),
    ("data", "attr_freq"), ("data", "plan_repair"), ("data", "dev_online"),
    ("trainer", "output_dir"), ("trainer", "deepspeed"), ("trainer", "logging_dir"),
}

# ----------------------------------------------------------------------------- defaults
_COMMON: dict[str, Any] = {
    "adapter": "qwen25_7b",
    "stage": "s1",
    "run_name": None,            # required (derived <adapter>_<stage> if absent)
    "init_from": None,           # s2: S1 best dir
    "data": {
        "index": "DATA/index_train.json",
        "meta": "DATA/index_meta.json",
        "ctx_meta": "DATA/index_ctx_meta.json",
        "motion_meta": "DATA/index_motion_meta.json",
        "attr_freq": "DATA/attr_class_freq.json",
        "plan_repair": "DATA/plan_repair.json",
        "dev_online": "DATA/dev_online_128.json",
        "long_edge": 2916,
        "attrqa_ratio": 0.15,
        "s2_partitions": list(P.S2_PARTITIONS),
        "point_mode_train": "random_interior",
    },
    "sampling": {
        "intent_power": 1.0, "context_power": 0.3, "context_cap": 5.0, "event_boost": 2.0,
        "object_cap": 4, "object_boost_ge3": 2.0, "combined_cap": 8.0,
        "motion_mult": {"start": 2.5, "stop": 2.0, "stay": 1.2, "decel": 1.2},
        "attr_mult": 1.5,
    },
    "loss": {"lambda_traj": 1.0, "field_weights": None},
    "model": {"attn": "flash_attention_2", "freeze_vision": False,
              "gradient_checkpointing": True, "visual_lr_mult": 0.1},
    # enabled: false OR every_steps <= 0 disables the online dev eval
    "eval": {"enabled": True, "every_steps": 500, "subset": 128, "max_new_tokens": 1024},
    "trainer": {
        "output_dir": None,      # -> RUNS/<run_name>
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "learning_rate": 1.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "num_train_epochs": 2,
        "bf16": True,
        "max_grad_norm": 1.0,
        "save_steps": 300,
        "save_total_limit": 3,
        "logging_steps": 10,
        "deepspeed": "PKG/configs/deepspeed/zero2.json",
        # operational TrainingArguments keys
        "save_strategy": "steps",
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,
        "ignore_data_skip": True,        # weighted sampling WITH replacement -> no data order to skip
        "gradient_checkpointing": False, # handled on the model in train_stage.py
        "report_to": "none",             # set to `wandb` in the yaml to log (sbatch exports WANDB_*)
        "run_name": None,                # -> run_name
        "seed": 42,
        # NCCL collective timeout. The dev-eval callback generates on rank 0 ONLY, so the other
        # ranks sit in an allreduce for the whole eval; the HF default of 1800 s aborts the run
        # as soon as a single eval pass takes longer than 30 minutes, which it easily does once
        # the curriculum unlocks the object fields and the generations get longer.
        "ddp_timeout": 14400,            # 4 h
    },
}
_STAGE_DEFAULTS: dict[str, dict[str, Any]] = {
    "s1": {"curriculum": {"schedule": ["a", "b"], "boundaries": [0.15]},
           "model": {"freeze_vision": False},
           "trainer": {"learning_rate": 1.0e-5}},
    "s2": {"curriculum": None,
           "model": {"freeze_vision": True},
           "trainer": {"learning_rate": 5.0e-6}},
}


# ----------------------------------------------------------------------------- helpers
def deep_merge(base: dict, override: Optional[dict]) -> dict:
    """Recursive merge: values in `override` win; an explicit None in `override` overrides
    (so `curriculum: null` disables the S1 default schedule). Returns a new dict."""
    out = copy.deepcopy(base)
    if not override:
        return out
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path_tokens(s: Any) -> Any:
    """Expand `${PKG}`-style and leading `PKG/`-style tokens (PKG ROOT RUNS DATA WEIGHTS CACHE EVAL LOGS)."""
    if not isinstance(s, str):
        return s
    for name, val in _PATH_TOKENS.items():
        s = s.replace("${" + name + "}", val)
        if s == name:
            s = val
        elif s.startswith(name + "/"):
            s = val + s[len(name):]
    return os.path.expanduser(s)


_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def _coerce_scalar(text: str) -> Any:
    """'1e-5' -> 1e-05, 'true' -> True, '[a,b]' -> ['a','b'], 'null' -> None, else str."""
    try:
        v = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    if isinstance(v, str) and _NUM_RE.match(v):   # yaml leaves '1e-5' (no dot) as a string
        return float(v)
    return v


def parse_dotlist(items: Optional[Iterable[str]]) -> dict:
    """['trainer.max_steps=3', 'eval.every_steps=0'] -> nested dict. Values parsed as YAML scalars."""
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"override must be key=value, got {it!r}")
        key, val = it.split("=", 1)
        d = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
            if not isinstance(d, dict):
                raise ValueError(f"override {it!r} conflicts with a scalar at {p!r}")
        d[parts[-1]] = _coerce_scalar(val.strip())
    return out


def default_config(stage: str = "s1") -> dict:
    """Documented defaults for `stage` (no file read)."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    cfg = deep_merge(_COMMON, _STAGE_DEFAULTS[stage])
    cfg["stage"] = stage
    return cfg


def _walk_paths(cfg: dict) -> None:
    for key in _PATH_KEYS:
        d = cfg
        for p in key[:-1]:
            d = d.get(p) if isinstance(d, dict) else None
            if d is None:
                break
        if isinstance(d, dict) and isinstance(d.get(key[-1]), str):
            d[key[-1]] = resolve_path_tokens(d[key[-1]])


def validate_config(cfg: dict) -> None:
    """Raise ValueError for an inconsistent config (does not touch the filesystem)."""
    if cfg.get("stage") not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {cfg.get('stage')!r}")
    if cfg.get("adapter") not in ADAPTERS:
        raise ValueError(f"adapter must be one of {ADAPTERS}, got {cfg.get('adapter')!r}")
    if not cfg.get("run_name"):
        raise ValueError("run_name is required")
    d = cfg["data"]
    if not (0.0 <= float(d["attrqa_ratio"]) <= 1.0):
        raise ValueError("data.attrqa_ratio must be in [0,1]")
    cur = cfg.get("curriculum")
    if cur is not None:
        sched, bounds = list(cur.get("schedule") or []), list(cur.get("boundaries") or [])
        if len(bounds) != max(len(sched) - 1, 0):
            raise ValueError(f"curriculum.boundaries must have len(schedule)-1 entries: {sched} {bounds}")
        if any(not (0.0 < float(b) < 1.0) for b in bounds):
            raise ValueError("curriculum.boundaries must lie in (0,1)")
        if cfg["stage"] == "s2":
            raise ValueError("S2 trains on the full field set and has no curriculum: set curriculum: null")
    if float(cfg["loss"]["lambda_traj"]) < 0:
        raise ValueError("loss.lambda_traj must be >= 0")
    fw = cfg["loss"].get("field_weights")
    if fw is not None and not isinstance(fw, dict):
        raise ValueError("loss.field_weights must be a mapping field -> weight (or null)")
    ev = cfg.get("eval")
    if ev is not None and not isinstance(ev.get("enabled", True), bool):
        raise ValueError("eval.enabled must be a boolean")
    tr = cfg["trainer"]
    for k in ("per_device_train_batch_size", "gradient_accumulation_steps", "save_steps", "logging_steps"):
        if int(tr[k]) <= 0:
            raise ValueError(f"trainer.{k} must be > 0")
    if not tr.get("output_dir"):
        raise ValueError("trainer.output_dir is empty")


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> dict:
    """Load `path` (yaml), fill defaults (stage-aware), apply dot-list `overrides`, resolve path
    tokens, derive run_name/output_dir/trainer.run_name and validate. Returns a plain dict."""
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    ov = parse_dotlist(overrides)
    stage = ov.get("stage") or user.get("stage") or _COMMON["stage"]
    cfg = deep_merge(default_config(stage), user)
    cfg = deep_merge(cfg, ov)
    cfg["stage"] = stage
    if not cfg.get("run_name"):
        cfg["run_name"] = f"{cfg['adapter']}_{stage}"
    tr = cfg["trainer"]
    if not tr.get("output_dir"):
        tr["output_dir"] = os.path.join(P.RUNS, cfg["run_name"])
    if not tr.get("run_name"):
        tr["run_name"] = cfg["run_name"]
    # numeric hygiene: yaml reads `1e-5` (no dot) as a string
    for k in ("learning_rate", "warmup_ratio", "max_grad_norm", "num_train_epochs", "weight_decay"):
        if k in tr and isinstance(tr[k], str):
            tr[k] = float(tr[k])
    cfg["loss"]["lambda_traj"] = float(cfg["loss"]["lambda_traj"])
    _walk_paths(cfg)
    if cfg.get("eval") is None:      # `eval: null` -> online dev eval off
        cfg["eval"] = {"enabled": False, "every_steps": 0, "subset": 0, "max_new_tokens": 1024}
    cfg["_config_path"] = os.path.abspath(path)
    validate_config(cfg)
    return cfg


def trainer_kwargs(cfg: dict) -> dict:
    """The `trainer` block with non-TrainingArguments keys removed (ready for TrainingArguments(**kw))."""
    kw = dict(cfg["trainer"])
    if kw.get("deepspeed") in ("", "null", "none", "None"):
        kw["deepspeed"] = None
    return kw


def to_yaml(cfg: dict) -> str:
    """Pretty yaml (private `_config_path` dropped)."""
    return yaml.safe_dump({k: v for k, v in cfg.items() if not k.startswith("_")},
                          sort_keys=False, allow_unicode=True, default_flow_style=None)


if __name__ == "__main__":  # `python -m training.core.config <yaml> [k=v ...]` -> resolved yaml
    import sys
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    c = load_config(sys.argv[1], sys.argv[2:])
    print(to_yaml(c))
