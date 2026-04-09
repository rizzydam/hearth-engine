"""
Information model — coverage, absence detection, confidence degradation.

This is what makes the system honest about what it doesn't know.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .contracts import CoverageReport, DataExpectation
from .network import SynapseNetwork, _domain_from_key
from .store import SynapseStore

log = logging.getLogger(__name__)


class InformationModel:
    """Tracks what data the system has, what it expects, and what's missing."""

    def __init__(self, network: SynapseNetwork, store: SynapseStore):
        self._network = network
        self._store = store
        self._expectations: dict[str, DataExpectation] = store.load_expectations()

    def update_expectation(self, metric_key: str, observation_time: str) -> None:
        """Record that we saw data for this metric. Updates cadence tracking."""
        exp = self._expectations.get(metric_key)
        if exp is None:
            # First observation — initialize with catalog cadence if available
            cadence = 24.0
            try:
                from ..metric_catalog.catalog import ALL_METRICS
                spec = ALL_METRICS.get(metric_key)
                if spec:
                    cadence = spec.expected_cadence_hours
            except ImportError:
                pass

            exp = DataExpectation(
                metric_key=metric_key,
                expected_cadence_hours=cadence,
                last_seen=observation_time,
                total_observations=1,
            )
        else:
            if exp.last_seen:
                # Compute gap and update running stats
                try:
                    last = datetime.fromisoformat(exp.last_seen)
                    now = datetime.fromisoformat(observation_time)
                    gap_hours = max(0, (now - last).total_seconds() / 3600)

                    n = exp.total_observations
                    old_avg = exp.avg_gap_hours
                    new_avg = old_avg + (gap_hours - old_avg) / (n + 1)
                    # Running variance (Welford's)
                    old_var = exp.gap_stddev_hours ** 2
                    new_var = old_var + ((gap_hours - old_avg) * (gap_hours - new_avg) - old_var) / (n + 1)
                    exp.avg_gap_hours = new_avg
                    exp.gap_stddev_hours = max(0, new_var) ** 0.5
                except (ValueError, TypeError):
                    pass

            exp.last_seen = observation_time
            exp.total_observations += 1
            exp.gap_count = 0
            exp.is_dropout = False

        self._expectations[metric_key] = exp

    def check_for_dropouts(self) -> list[DataExpectation]:
        """Check all expectations for metrics that have gone silent."""
        now = datetime.now(timezone.utc)
        dropouts = []

        for key, exp in self._expectations.items():
            if not exp.last_seen:
                continue
            try:
                last = datetime.fromisoformat(exp.last_seen)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                hours_since = (now - last).total_seconds() / 3600
            except (ValueError, TypeError):
                continue

            expected_gaps = hours_since / max(exp.expected_cadence_hours, 1)
            if expected_gaps > 1.5:  # 1.5x expected cadence = potential gap
                exp.gap_count = int(expected_gaps)
                if exp.gap_stddev_hours > 0 and exp.total_observations > 10:
                    exp.gap_z_score = (hours_since - exp.avg_gap_hours) / exp.gap_stddev_hours
                    exp.is_dropout = exp.gap_z_score > 2.0
                else:
                    exp.gap_z_score = 0.0
                    exp.is_dropout = False

                if exp.gap_count >= 2:
                    dropouts.append(exp)

        return dropouts

    def compute_coverage(self, current_values: dict[str, float]) -> CoverageReport:
        """How much of the network can we see right now?"""
        total = self._network.count()
        visible = 0
        partial = 0
        blind = 0
        blind_weights: list[tuple[str, float]] = []

        for syn in self._network.all():
            a_present = syn.metric_a in current_values
            b_present = syn.metric_b in current_values
            if a_present and b_present:
                visible += 1
            elif a_present or b_present:
                partial += 1
            else:
                blind += 1
                blind_weights.append((syn.synapse_id, syn.weight))

        # Highest-value blind synapses — where data would help most
        blind_weights.sort(key=lambda x: x[1], reverse=True)
        highest_blind = [sid for sid, _ in blind_weights[:10]]

        # Per-domain coverage
        per_domain: dict[str, tuple[int, int]] = {}
        for syn in self._network.all():
            for metric in (syn.metric_a, syn.metric_b):
                domain = _domain_from_key(metric)
                total_d, visible_d = per_domain.get(domain, (0, 0))
                per_domain[domain] = (total_d + 1, visible_d + (1 if metric in current_values else 0))

        domain_coverage = {
            d: v / max(t, 1) for d, (t, v) in per_domain.items()
        }

        return CoverageReport(
            total_synapses=total,
            visible_synapses=visible,
            partially_visible=partial,
            blind_synapses=blind,
            coverage_fraction=visible / max(total, 1),
            highest_value_blind=highest_blind,
            per_domain_coverage=domain_coverage,
        )

    def marginal_value_requests(
        self, current_values: dict[str, float],
    ) -> list[dict]:
        """Which missing metrics would illuminate the most synapses?"""
        present = set(current_values.keys())
        all_metrics = self._network.all_metrics()
        missing = all_metrics - present

        requests = []
        for metric in missing:
            connected = self._network.get_connected(metric)
            total_weight = sum(s.weight for s in connected)
            high_weight = sum(1 for s in connected if s.weight > 0.3)

            if connected:
                requests.append({
                    "metric_key": metric,
                    "reason": f"Would illuminate {len(connected)} synapses "
                              f"({high_weight} high-weight)",
                    "marginal_value": round(total_weight, 3),
                    "connected_count": len(connected),
                })

        requests.sort(key=lambda r: r["marginal_value"], reverse=True)
        return requests[:10]

    def save(self) -> None:
        self._store.save_expectations(self._expectations)
