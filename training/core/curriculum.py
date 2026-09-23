#!/usr/bin/env python3
"""Stage-1 curriculum: field-unlock sets + the Trainer callback that switches them.

Two stages: ``a`` covers the first 15 % of training and supervises the scene-level fields
only; ``b`` covers the remaining 85 % and additionally unlocks the per-object fields.

The curriculum only changes WHICH fields contribute to the loss (``struct`` tokens are
always supervised so the model learns the format); model freezing and the LR schedule stay
constant across stages (single cosine schedule over the whole run).

Stage state lives in a shared ``multiprocessing.Value`` inside the dataset, so forked
dataloader workers -- and the ``StageAwareWeightedSampler`` -- pick up the switch too.
"""
from __future__ import annotations

from typing import Iterable, Optional, Set

from transformers import TrainerCallback

# ``None`` would mean "all fields" (used when there is no schedule).
CURRICULUM_STAGES: dict[str, Optional[Set[str]]] = {
    "a": {"context", "events", "presence"},
    "b": {"context", "events", "presence", "count", "obj_open", "type", "point", "rank",
          "location", "intention", "state", "content"},
}

DEFAULT_SCHEDULE = ["a", "b"]
DEFAULT_BOUNDARIES = [0.15]


def unlock_for(stage_name: Optional[str]) -> Optional[Set[str]]:
    """Field set unlocked in curriculum stage ``stage_name`` (``None``/unknown -> all fields)."""
    if stage_name is None:
        return None
    return CURRICULUM_STAGES.get(stage_name)


def stage_index_for(frac: float, boundaries: Iterable[float]) -> int:
    """Index into the schedule for training progress ``frac`` in [0, 1]."""
    idx = 0
    for b in boundaries:
        if frac >= b:
            idx += 1
        else:
            break
    return idx


class CurriculumCallback(TrainerCallback):
    """Switch ``train_dataset``'s curriculum stage at fractions of ``state.max_steps``.

    ``schedule``: e.g. ``['a', 'b']``; ``boundaries``: cumulative step fractions between
    stages, e.g. ``[0.15]`` -> a: [0, 15 %), b: [15 %, 100 %]. ``len(boundaries)`` must equal
    ``len(schedule) - 1``. The dataset must expose ``set_stage(idx)`` (``WaymoDataset``).
    """

    def __init__(self, train_dataset, schedule=DEFAULT_SCHEDULE, boundaries=DEFAULT_BOUNDARIES):
        self.ds = train_dataset
        self.schedule = list(schedule)
        self.boundaries = [float(b) for b in boundaries]
        if len(self.boundaries) != len(self.schedule) - 1:
            raise ValueError("need len(schedule)-1 boundaries "
                             f"(schedule={self.schedule}, boundaries={self.boundaries})")
        self._cur = -1

    def _stage_for(self, frac: float) -> int:
        return stage_index_for(frac, self.boundaries)

    def current_index(self) -> int:
        """Stage index most recently applied to the dataset (-1 before ``on_train_begin``)."""
        return self._cur

    def _maybe_switch(self, state):
        total = state.max_steps or 1
        idx = self._stage_for(state.global_step / total)
        if idx != self._cur:
            self._cur = idx
            self.ds.set_stage(idx)
            if getattr(state, "is_world_process_zero", True):
                print(f"[curriculum] step {state.global_step}/{total} "
                      f"-> stage {idx} ('{self.schedule[idx]}')")

    def on_train_begin(self, args, state, control, **kw):
        # Resumed runs: pick the stage matching global_step instead of blindly resetting to 0.
        self._cur = -1
        self._maybe_switch(state)
        if getattr(state, "is_world_process_zero", True):
            print(f"[curriculum] schedule={self.schedule} boundaries={self.boundaries}")

    def on_step_begin(self, args, state, control, **kw):
        self._maybe_switch(state)
