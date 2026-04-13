"""
Recovery state machine -- tracks where someone is in the recovery sequence.

sleep -> activity -> social -> work -> financial -> identity

This isn't prescriptive. It's descriptive. It reads the data and says
"based on what's improving, you're at the activity stage of recovery."
"""

from __future__ import annotations

from .contracts import RecoveryState, RECOVERY_STAGES
from .store import SynapseStore


# Map each recovery stage to the metric that gates advancement.
# These are example mappings -- override with your domain's metric keys.
STAGE_METRICS: dict[str, str] = {
    "sleep": "fitness.sleep_hours",
    "activity": "fitness.exercise_minutes",
    "social": "project.team_velocity",
    "work": "project.task_completion_rate",
    "financial": "business.monthly_revenue",
    "identity": "business.client_satisfaction",
}


class RecoveryStateMachine:
    """Tracks recovery trajectory through the cascade recovery sequence."""

    def __init__(self, store: SynapseStore):
        self._store = store
        self._state = store.load_recovery_state()

    @property
    def state(self) -> RecoveryState:
        return self._state

    def update(
        self,
        current_values: dict[str, float],
        domain_severities: dict[str, float],
    ) -> RecoveryState:
        """Update recovery state based on current metric values.

        Reads the gating metric for the current stage and decides:
        - Advance: gating metric is improving/thriving → progress++
        - Hold: gating metric is stable/healthy → no change
        - Regress: gating metric is declining/warning → progress-- or stage--
        """
        # Determine overall trajectory from domain severities
        severe_count = sum(1 for v in domain_severities.values() if v > 0.5)
        improving_count = sum(1 for v in domain_severities.values() if v < 0.1)

        if severe_count >= 3:
            self._state.trajectory = "deteriorating"
        elif severe_count >= 1:
            self._state.trajectory = "stabilizing"
        elif improving_count >= 4:
            self._state.trajectory = "recovered"
        else:
            self._state.trajectory = "recovering"

        # If in crisis/deteriorating, reset to stage 0
        if self._state.trajectory == "deteriorating" and self._state.stage_index > 0:
            self._record_exit("regressed_to_start")
            self._state.stage_index = 0
            self._state.stage_progress = 0.0
            self._state.regression_count += 1

        # Check current stage's gating metric
        stage_name = RECOVERY_STAGES[self._state.stage_index]
        self._state.current_stage = stage_name
        gating_metric = STAGE_METRICS.get(stage_name)

        if gating_metric and gating_metric in current_values:
            value = current_values[gating_metric]
            level = self._evaluate(gating_metric, value)

            if level == "thriving":
                self._state.stage_progress = min(1.0, self._state.stage_progress + 0.2)
            elif level == "healthy":
                self._state.stage_progress = min(1.0, self._state.stage_progress + 0.05)
            elif level == "warning":
                self._state.stage_progress = max(0.0, self._state.stage_progress - 0.1)
            elif level == "critical":
                self._state.stage_progress = max(0.0, self._state.stage_progress - 0.2)
                if self._state.stage_progress <= 0 and self._state.stage_index > 0:
                    self._record_exit("regressed")
                    self._state.stage_index -= 1
                    self._state.stage_progress = 0.5
                    self._state.regression_count += 1

            # Advance to next stage when progress reaches 1.0
            if self._state.stage_progress >= 1.0 and self._state.stage_index < len(RECOVERY_STAGES) - 1:
                self._record_exit("advanced")
                self._state.stage_index += 1
                self._state.stage_progress = 0.0

        self._state.current_stage = RECOVERY_STAGES[self._state.stage_index]
        self._state.time_in_stage_days += 1

        self._store.save_recovery_state(self._state)
        return self._state

    def _record_exit(self, outcome: str) -> None:
        from .contracts import _now_iso
        self._state.history.append({
            "stage": self._state.current_stage,
            "outcome": outcome,
            "exited_at": _now_iso(),
            "days_in_stage": self._state.time_in_stage_days,
        })
        self._state.time_in_stage_days = 0
        # Keep history bounded
        if len(self._state.history) > 50:
            self._state.history = self._state.history[-50:]

    def _evaluate(self, metric_key: str, value: float) -> str:
        try:
            from ..metric_catalog.catalog import ALL_METRICS
            spec = ALL_METRICS.get(metric_key)
            if spec:
                return spec.evaluate(value)
        except ImportError:
            pass
        return "healthy"
