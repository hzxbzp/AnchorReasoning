"""Fairness layer for the released ('domain-adapted', zero-shot) models.

The released checkpoints differ in what they can natively produce (per-object image points,
object types, a plan, a reason, a trajectory).  This package scores each one on exactly the
capabilities it has and marks the rest '—', instead of charging a model for a format it was never
trained on.  It is ADDITIVE: it reuses the shared scoring code in ``evaluation.eval_dev`` and
``training.core.metrics`` without modifying it.

    profiles.py      capability profile per model (what to score, where the outputs come from)
    text_extract.py  tolerant free text -> parse_output-shaped dict
    typeonly.py      geometry-free L1/L2 for models that emit no points
    rescore.py       chains.json -> metrics_baseline.json (capability-aware, judges, RFS)
"""
