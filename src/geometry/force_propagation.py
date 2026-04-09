"""
Force propagation — synapse coupling through iterative relaxation.

Each axis pulls connected axes proportional to synapse weight × deviation.
Forces propagate until the shape stabilizes. This is the heart of the
geometric model: metrics don't exist in isolation.

The result: a coupled shape where sleep dropping to 0.6 doesn't just
affect the sleep axis — it pulls mental health inward, which pulls
financial decision-making inward, which pulls career inward. The
geometry captures the compound effect that a list of numbers cannot.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Damping factor: each iteration applies this fraction of computed force.
# 0.3 ensures convergence in 3-5 iterations without oscillation.
DAMPING = 0.3

# Maximum iterations before giving up
MAX_ITERATIONS = 5

# Convergence threshold: stop when max force < this
CONVERGENCE_THRESHOLD = 0.01

# Value clamp range
MIN_VALUE = 0.0
MAX_VALUE = 2.5


def propagate(
    normalized_values: dict[str, float],
    network: Any,
    hard_rule_floors: dict[str, float] | None = None,
) -> tuple[dict[str, float], int, float]:
    """Run iterative force propagation through the synapse network.

    Parameters
    ----------
    normalized_values : dict[str, float]
        Baseline-normalized values centered at 1.0.
    network : SynapseNetwork
        The synapse graph to traverse.
    hard_rule_floors : dict[str, float], optional
        Minimum values enforced by hard rules (can't be overridden by
        positive forces from other domains).

    Returns
    -------
    coupled_values : dict[str, float]
        Values after force propagation.
    iterations : int
        How many iterations were needed.
    residual : float
        Maximum force magnitude at convergence.
    """
    coupled = dict(normalized_values)
    floors = hard_rule_floors or {}

    all_synapses = network.all()
    if not all_synapses:
        return coupled, 0, 0.0

    iterations = 0
    residual = 0.0

    for iteration in range(MAX_ITERATIONS):
        forces: dict[str, float] = {}
        max_force = 0.0

        for syn in all_synapses:
            source_val = coupled.get(syn.metric_a)
            if source_val is None:
                continue

            deviation = source_val - 1.0
            if abs(deviation) < 0.03:
                continue  # Too small to propagate

            # Apply non-linear activation if specified (step, sigmoid, cliff)
            act_fn = syn.metadata.get("activation_function", "linear") if syn.metadata else "linear"
            if act_fn and act_fn != "linear":
                from ..synapse.activation import compute_activation
                act_params = syn.metadata.get("activation_params", {}) if syn.metadata else {}
                deviation = compute_activation(deviation, act_fn, act_params)
                if abs(deviation) < 0.01:
                    continue  # Activation suppressed this synapse

            # Force magnitude = deviation × weight × confidence × damping
            base_force = deviation * syn.weight * syn.confidence * DAMPING

            # Direction depends on relationship type
            if syn.relationship == "threatens":
                # Source degraded → push target AWAY from 1.0 (same direction)
                # Sleep drops below 1.0 → mental gets pulled below 1.0
                force = base_force
            elif syn.relationship == "protects":
                # Source strong → push target TOWARD 1.0 (dampens deviation)
                # Sleep above 1.0 → mental gets pulled toward 1.0 from wherever it is
                target_val = coupled.get(syn.metric_b, 1.0)
                target_deviation = target_val - 1.0
                if abs(target_deviation) < 0.03:
                    continue
                # Protective force: push target toward 1.0
                # Magnitude proportional to source strength AND target deviation
                force = -target_deviation * syn.weight * syn.confidence * DAMPING * 0.5
            elif syn.relationship == "reinforces":
                # Source thriving → push target ABOVE 1.0
                # Only applies when source is above baseline
                if deviation <= 0:
                    continue  # Reinforcement only works when source is strong
                force = deviation * syn.weight * syn.confidence * DAMPING * 0.7
            else:
                force = base_force * 0.5  # Unknown relationship type

            # Accumulate forces on the target metric
            forces[syn.metric_b] = forces.get(syn.metric_b, 0.0) + force
            max_force = max(max_force, abs(force))

        # Apply accumulated forces
        for key, force in forces.items():
            if key in coupled:
                coupled[key] = coupled[key] + force
            else:
                coupled[key] = 1.0 + force

            # Clamp
            coupled[key] = max(MIN_VALUE, min(MAX_VALUE, coupled[key]))

        # Enforce hard rule floors
        for key, floor_val in floors.items():
            if key in coupled:
                coupled[key] = min(coupled[key], floor_val)

        iterations = iteration + 1
        residual = max_force

        if max_force < CONVERGENCE_THRESHOLD:
            break

    return coupled, iterations, residual
