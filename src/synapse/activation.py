"""
Non-linear activation functions for synapse force propagation.

Linear propagation is wrong for certain patterns:
- The freeze: below threshold = functional, above = total shutdown (step)
- The illusion mechanic: gradual onset, rapid cascade, saturation (sigmoid)
- All-or-nothing: fine above threshold, instant collapse below (cliff)

These functions transform the raw metric deviation before it becomes
force in the propagation loop.
"""

from __future__ import annotations

import math


def compute_activation(
    deviation: float,
    activation_function: str,
    params: dict,
) -> float:
    """Apply a non-linear activation function to a metric deviation.

    Parameters
    ----------
    deviation : float
        The raw deviation from baseline (value - 1.0 in normalized space).
    activation_function : str
        One of: "linear", "step", "sigmoid", "cliff"
    params : dict
        Function-specific parameters (threshold, midpoint, steepness, etc.)

    Returns
    -------
    float
        Transformed deviation. Same sign convention as input.
    """
    if activation_function == "linear" or not activation_function:
        return deviation

    sign = 1.0 if deviation >= 0 else -1.0
    magnitude = abs(deviation)

    if activation_function == "step":
        # Binary: below threshold = 0, above = full magnitude
        threshold = params.get("threshold", 0.5)
        if magnitude < threshold:
            return 0.0
        return sign * magnitude

    elif activation_function == "sigmoid":
        # S-curve: slow onset, rapid middle, saturation
        midpoint = params.get("midpoint", 0.5)
        steepness = params.get("steepness", 8.0)
        sigmoid_val = 1.0 / (1.0 + math.exp(-steepness * (magnitude - midpoint)))
        return sign * sigmoid_val * magnitude

    elif activation_function == "cliff":
        # Fine above threshold, instant collapse below
        threshold = params.get("threshold", 0.3)
        if magnitude < threshold:
            return sign * magnitude * 0.1  # Mild effect above cliff
        return sign * magnitude  # Full effect at/below cliff

    return deviation
