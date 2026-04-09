"""
Hebbian learning engine — connections that fire together wire together.

When metrics co-move within a synapse's lag window, the connection
strengthens. When expected co-movement doesn't occur, it weakens.
Without reinforcement, everything decays toward zero.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .contracts import Synapse, MetricObservation, _new_id, _now_iso
from .network import SynapseNetwork
from .store import SynapseStore

log = logging.getLogger(__name__)


class HebbianLearner:
    """Learns synapse weights from observation data."""

    def __init__(self, network: SynapseNetwork, store: SynapseStore):
        self._network = network
        self._store = store

    def process_observations(self, observations: list[MetricObservation]) -> int:
        """Process new observations and update synapse weights.

        For each new observation, check if any synapse involving this metric
        has recently seen activity on the other end. If so, reinforce or weaken.

        Returns count of synapses modified.
        """
        if not observations:
            return 0

        # Store observations
        self._store.append_observations(observations)

        # Group new observations by metric
        new_by_metric: dict[str, list[MetricObservation]] = {}
        for obs in observations:
            new_by_metric.setdefault(obs.metric_key, []).append(obs)

        modified = 0
        for metric_key, new_obs in new_by_metric.items():
            connected = self._network.get_connected(metric_key)
            for syn in connected:
                other_key = syn.metric_b if syn.metric_a == metric_key else syn.metric_a

                # Check if we have recent observations for the other metric
                other_obs = self._store.load_observations(other_key)
                if len(other_obs) < 5:
                    continue

                # Compute correlation between the two metrics
                my_obs = self._store.load_observations(metric_key)
                if len(my_obs) < 5:
                    continue

                r = self._compute_correlation(my_obs, other_obs)
                if r is not None:
                    self._reinforce(syn, r)
                    modified += 1

        return modified

    def observe_and_discover(
        self,
        observations: list[MetricObservation],
        min_correlation: float = 0.3,
        min_observations: int = 14,
    ) -> int:
        """Process observations AND discover new synapses from co-movement.

        This is the full learning cycle: reinforce existing + discover new.
        """
        self._store.append_observations(observations)

        # Get all metrics with enough history
        all_obs = self._store.load_observations()
        by_metric: dict[str, list[MetricObservation]] = {}
        for obs in all_obs:
            by_metric.setdefault(obs.metric_key, []).append(obs)

        metrics_with_data = [k for k, v in by_metric.items() if len(v) >= min_observations]
        created = 0

        # Check all pairs of metrics with sufficient data
        for i, key_a in enumerate(metrics_with_data):
            for key_b in metrics_with_data[i + 1:]:
                existing = self._network.find_existing(key_a, key_b)

                r = self._compute_correlation(by_metric[key_a], by_metric[key_b])
                if r is None:
                    continue

                if existing:
                    self._reinforce(existing, r)
                elif abs(r) >= min_correlation:
                    # Discover new synapse
                    syn = Synapse(
                        synapse_id=_new_id("obs"),
                        metric_a=key_a,
                        metric_b=key_b,
                        weight=round(abs(r) * 0.5, 3),  # Conservative start
                        direction="positive" if r > 0 else "negative",
                        relationship="reinforces" if r > 0 else "threatens",
                        source="observed",
                        decay_rate=0.015,
                    )
                    self._network.add_synapse(syn)
                    created += 1
                    self._log_event(syn.synapse_id, "created", r)

        return created

    def _reinforce(self, syn: Synapse, observed_r: float) -> None:
        """Reinforce or weaken a synapse based on observed correlation."""
        delta = 0.02 * abs(observed_r)

        # Does the observation match the synapse's expected direction?
        if (observed_r > 0 and syn.direction == "positive") or \
           (observed_r < 0 and syn.direction == "negative"):
            # Confirming
            syn.weight = min(1.0, syn.weight + delta)
            syn.observations += 1
            if syn.source == "research_prior" and not syn.user_confirmed:
                syn.user_confirmed = True
                self._log_event(syn.synapse_id, "user_confirmed", observed_r)
        else:
            # Contradicting — weaken slower than strengthen
            syn.weight = max(0.0, syn.weight - delta * 0.5)
            syn.contradictions += 1

        # Update confidence
        total = syn.observations + syn.contradictions
        syn.confidence = syn.observations / (total + 1) if total > 0 else 0.5
        syn.last_reinforced = _now_iso()

    def anti_reinforce(self, synapse_id: str) -> None:
        """Called when expected co-movement did NOT occur.

        metric_a changed significantly but metric_b didn't, within the
        lag window. The synapse predicted they should co-move. They didn't.
        """
        syn = self._network.get(synapse_id)
        if syn is None:
            return
        syn.weight = max(0.0, syn.weight - 0.01)
        syn.contradictions += 1
        total = syn.observations + syn.contradictions
        syn.confidence = syn.observations / (total + 1) if total > 0 else 0.5

    def decay_all(self) -> int:
        """Apply daily decay to all synapses. Returns count of decayed."""
        now = datetime.now(timezone.utc)
        decayed = 0

        for syn in self._network.all():
            if not syn.last_reinforced:
                continue
            try:
                last = datetime.fromisoformat(syn.last_reinforced)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                days_since = max(0, (now - last).total_seconds() / 86400)
            except (ValueError, TypeError):
                continue

            if days_since < 1:
                continue

            decay = syn.decay_rate * days_since
            old_weight = syn.weight
            syn.weight = max(0.0, syn.weight - decay)

            if syn.weight < old_weight:
                decayed += 1

        return decayed

    def prune(self, weight_threshold: float = 0.05, min_observations: int = 3) -> int:
        """Remove synapses that have decayed below threshold.

        Research priors are never auto-pruned — they can decay to near-zero
        but remain as dormant connections.
        """
        to_remove = []
        for syn in self._network.all():
            if syn.source == "research_prior":
                continue  # Never prune research priors
            if syn.weight < weight_threshold and syn.observations < min_observations:
                to_remove.append(syn.synapse_id)

        for sid in to_remove:
            self._network.remove_synapse(sid)
            self._log_event(sid, "pruned", 0.0)

        return len(to_remove)

    def _compute_correlation(
        self,
        obs_a: list[MetricObservation],
        obs_b: list[MetricObservation],
    ) -> float | None:
        """Compute Pearson correlation between two observation series."""
        try:
            from ..compound_intelligence import compute_correlation
        except ImportError:
            return self._simple_correlation(obs_a, obs_b)

        # Align by date (both series need matching timestamps)
        a_by_date: dict[str, float] = {}
        for obs in obs_a:
            date = obs.timestamp[:10]  # YYYY-MM-DD
            a_by_date[date] = obs.value

        b_by_date: dict[str, float] = {}
        for obs in obs_b:
            date = obs.timestamp[:10]
            b_by_date[date] = obs.value

        common_dates = sorted(set(a_by_date.keys()) & set(b_by_date.keys()))
        if len(common_dates) < 5:
            return None

        values_a = [a_by_date[d] for d in common_dates]
        values_b = [b_by_date[d] for d in common_dates]

        return compute_correlation(values_a, values_b)

    @staticmethod
    def _simple_correlation(
        obs_a: list[MetricObservation],
        obs_b: list[MetricObservation],
    ) -> float | None:
        """Fallback stdlib-only Pearson correlation."""
        a_by_date = {o.timestamp[:10]: o.value for o in obs_a}
        b_by_date = {o.timestamp[:10]: o.value for o in obs_b}
        common = sorted(set(a_by_date) & set(b_by_date))
        if len(common) < 5:
            return None

        a = [a_by_date[d] for d in common]
        b = [b_by_date[d] for d in common]
        n = len(a)
        mean_a = sum(a) / n
        mean_b = sum(b) / n
        cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n)) / n
        std_a = (sum((x - mean_a) ** 2 for x in a) / n) ** 0.5
        std_b = (sum((x - mean_b) ** 2 for x in b) / n) ** 0.5
        if std_a == 0 or std_b == 0:
            return None
        return max(-1.0, min(1.0, cov / (std_a * std_b)))

    def test_hypotheses(
        self,
        current_values: dict[str, float],
        registry: "HypothesisRegistry",
    ) -> int:
        """Test narrative hypotheses against current metric values.

        For each hypothesis with status untested/testing:
        - Check if the condition metric has a value
        - Check if the target metric has a value
        - Compare predicted direction with actual co-movement
        - Update predictions_correct/wrong and confidence

        Returns count of hypotheses tested this cycle.
        """
        tested = 0
        for h in registry.all():
            if h.status == "falsified":
                continue

            val_a = current_values.get(h.metric_a)
            val_b = current_values.get(h.metric_b)
            if val_a is None or val_b is None:
                continue

            # Find objective synapse to compare against
            syn = self._network.find_existing(h.metric_a, h.metric_b)
            if syn is None:
                continue

            # Can we evaluate? Need the synapse to have been reinforced/weakened
            if syn.observations + syn.contradictions < 3:
                if h.status == "untested":
                    h.status = "testing"
                continue

            # Compare: does the objective data confirm the hypothesis direction?
            # The synapse's observation/contradiction ratio tells us
            obj_confidence = syn.observations / max(syn.observations + syn.contradictions, 1)

            # The hypothesis predicts a specific weight_modifier
            # If modifier > 1.0, hypothesis says this connection is STRONGER than average
            # Check if the objective evidence supports "stronger"
            predicted_strong = h.weight_modifier > 1.0
            is_strong = obj_confidence > 0.6 and syn.weight > 0.3

            if (predicted_strong and is_strong) or (not predicted_strong and not is_strong):
                h.predictions_correct += 1
            else:
                h.predictions_wrong += 1

            h.predictions_made = h.predictions_correct + h.predictions_wrong
            h.confidence = h.predictions_correct / max(h.predictions_made, 1)
            h.last_tested = _now_iso()

            # Status transitions
            if h.predictions_made >= 10:
                if h.confidence > 0.6:
                    h.status = "validated"
                elif h.confidence < 0.3:
                    h.status = "falsified"
                else:
                    h.status = "testing"
            elif h.status == "untested":
                h.status = "testing"

            tested += 1

        if tested > 0:
            registry.save()

        return tested

    def _log_event(self, synapse_id: str, action: str, value: float) -> None:
        self._store.append_learning_event({
            "synapse_id": synapse_id,
            "action": action,
            "value": round(value, 4),
            "timestamp": _now_iso(),
        })
