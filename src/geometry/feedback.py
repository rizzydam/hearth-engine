"""
Feedback engine — closes the loop between recommendations and outcomes.

When the system recommends something and the user acts on it, the
outcome (improved/not improved) feeds back into synapse weights.
This is how the system learns what works for THIS person.
"""

from __future__ import annotations

import logging
from typing import Any

from .contracts import ShapeSnapshot

log = logging.getLogger(__name__)


class FeedbackEngine:
    """Process action outcomes and update synapse weights."""

    def __init__(self, synapse_engine: Any):
        self._synapse = synapse_engine

    def process_action_outcome(
        self,
        action_domain: str,
        acted_on: bool,
        pre_shape: ShapeSnapshot | None,
        post_shape: ShapeSnapshot | None,
    ) -> None:
        """Feed action outcome back into the synapse network.

        Parameters
        ----------
        action_domain : str
            The domain the recommendation targeted.
        acted_on : bool
            Whether the user acted on the recommendation.
        pre_shape : ShapeSnapshot
            Shape when the recommendation was made.
        post_shape : ShapeSnapshot
            Shape after the action period (or after ignoring).
        """
        if pre_shape is None or post_shape is None:
            return

        network = self._synapse.network

        pre_domain_val = pre_shape.per_domain_means.get(action_domain, 1.0)
        post_domain_val = post_shape.per_domain_means.get(action_domain, 1.0)
        improvement = post_domain_val - pre_domain_val

        if acted_on:
            if improvement > 0.05:
                # Acted + improved: strengthen domain synapses
                self._adjust_domain_synapses(network, action_domain, delta=0.03)
                log.info("Feedback: %s acted+improved (+%.3f), strengthening", action_domain, improvement)
            elif improvement < -0.05:
                # Acted but got worse: slightly weaken
                self._adjust_domain_synapses(network, action_domain, delta=-0.01)
                log.info("Feedback: %s acted+worsened (%.3f), weakening", action_domain, improvement)
        else:
            # Ignored: slightly weaken the connections that produced the rec
            self._adjust_domain_synapses(network, action_domain, delta=-0.005)

        network.save()

    def _adjust_domain_synapses(self, network: Any, domain: str, delta: float) -> None:
        """Adjust weights of all synapses targeting a domain."""
        for syn in network.all():
            target_domain = _domain_from_metric(syn.metric_b)
            if target_domain == domain:
                syn.weight = max(0.0, min(1.0, syn.weight + delta))


def _domain_from_metric(metric_key: str) -> str:
    try:
        from ..metric_catalog.catalog import ALL_METRICS, ALL_SENSES
        spec = ALL_METRICS.get(metric_key)
        if spec:
            return ALL_SENSES[spec.sense].domain
    except ImportError:
        pass
    return metric_key.split(".")[0].replace("sense", "")
