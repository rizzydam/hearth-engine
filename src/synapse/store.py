"""
SynapseStore — persistence for the synapse network.

JSON-first using LocalStore. Each collection maps to a future SQLite table.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from .contracts import (
    Synapse, MetricObservation, DataExpectation, RecoveryState,
)

log = logging.getLogger(__name__)

# How long to keep rolling windows
OBSERVATION_WINDOW_DAYS = 90
ASSESSMENT_HISTORY_LIMIT = 30
LEARNING_LOG_LIMIT = 1000


class SynapseStore:
    """Persistence layer wrapping LocalStore for synapse data."""

    def __init__(self, base_dir: Path | None = None):
        from ..storage import LocalStore  # type: ignore[import]  # optional dependency
        if base_dir is None:
            base_dir = Path.home() / ".home" / "synapse"
        self._store = LocalStore("synapse", base_dir=base_dir)

    # ── Synapses ──────────────────────────────────────────────────────

    def load_synapses(self) -> list[Synapse]:
        records = self._store.load("synapses")
        synapses = []
        for r in records:
            try:
                synapses.append(Synapse.from_dict(r))
            except Exception as e:
                log.warning("Skipping malformed synapse: %s", e)
        return synapses

    def save_synapses(self, synapses: list[Synapse]) -> None:
        self._store.save("synapses", [s.to_dict() for s in synapses])

    # ── Observations ──────────────────────────────────────────────────

    def load_observations(self, metric_key: str | None = None) -> list[MetricObservation]:
        records = self._store.load("observations")
        obs = []
        for r in records:
            try:
                o = MetricObservation.from_dict(r)
                if metric_key is None or o.metric_key == metric_key:
                    obs.append(o)
            except Exception:
                pass
        return obs

    def append_observations(self, observations: list[MetricObservation]) -> None:
        existing = self._store.load("observations")
        existing.extend([o.to_dict() for o in observations])

        # Prune to window
        cutoff = (datetime.now(timezone.utc) - timedelta(days=OBSERVATION_WINDOW_DAYS)).isoformat()
        pruned = [r for r in existing if r.get("timestamp", "") >= cutoff]
        self._store.save("observations", pruned)

    # ── Data Expectations ─────────────────────────────────────────────

    def load_expectations(self) -> dict[str, DataExpectation]:
        records = self._store.load("expectations")
        result = {}
        for r in records:
            try:
                exp = DataExpectation.from_dict(r)
                result[exp.metric_key] = exp
            except Exception:
                pass
        return result

    def save_expectations(self, expectations: dict[str, DataExpectation]) -> None:
        self._store.save("expectations", [e.to_dict() for e in expectations.values()])

    # ── Recovery State ────────────────────────────────────────────────

    def load_recovery_state(self) -> RecoveryState:
        records = self._store.load("recovery_state")
        if records:
            try:
                return RecoveryState.from_dict(records[0])
            except Exception:
                pass
        return RecoveryState()

    def save_recovery_state(self, state: RecoveryState) -> None:
        self._store.save("recovery_state", [state.to_dict()])

    # ── Assessment History ────────────────────────────────────────────

    def append_assessment(self, assessment_dict: dict[str, Any]) -> None:
        existing = self._store.load("assessment_history")
        existing.append(assessment_dict)
        self._store.save("assessment_history", existing[-ASSESSMENT_HISTORY_LIMIT:])

    def load_assessment_history(self) -> list[dict]:
        return self._store.load("assessment_history")

    # ── Learning Log ──────────────────────────────────────────────────

    def append_learning_event(self, event: dict[str, Any]) -> None:
        existing = self._store.load("learning_log")
        existing.append(event)
        self._store.save("learning_log", existing[-LEARNING_LOG_LIMIT:])
