"""
Shape metrics — geometric properties computed from coupled axis values.

The shape IS the assessment. These numbers tell you everything:
- area_ratio: overall contraction/expansion (health/severity)
- circularity: balance across domains (lopsided = bad even if average is ok)
- dominant_distortion: where the shape deviates most from the circle
- per_domain_means: per-domain health as shape property
"""

from __future__ import annotations

import math
from collections import defaultdict

from .contracts import AxisValue, ShapeSnapshot


def compute_shape(
    axes: list[AxisValue],
) -> ShapeSnapshot:
    """Compute all shape metrics from a list of axis values.

    Uses the `coupled` field of each axis (post-force-propagation).
    """
    if not axes:
        return ShapeSnapshot()

    # Sort axes by angle for polygon computation
    sorted_axes = sorted(axes, key=lambda a: a.angle_radians)

    # Compute vertex coordinates (polar → cartesian)
    vertices = []
    for ax in sorted_axes:
        x = ax.coupled * math.cos(ax.angle_radians)
        y = ax.coupled * math.sin(ax.angle_radians)
        vertices.append((x, y))

    # Area via shoelace formula
    area = _polygon_area(vertices)
    ideal_area = math.pi  # Area of unit circle (r=1.0)
    area_ratio = area / ideal_area if ideal_area > 0 else 1.0

    # Circularity: 4π × area / perimeter²
    perimeter = _polygon_perimeter(vertices)
    if perimeter > 0:
        circularity = (4 * math.pi * area) / (perimeter * perimeter)
        circularity = max(0.0, min(1.0, circularity))
    else:
        circularity = 1.0

    # Per-domain means and balance (variance within domain)
    domain_values: dict[str, list[float]] = defaultdict(list)
    for ax in sorted_axes:
        domain_values[ax.domain].append(ax.coupled)

    per_domain_means = {}
    per_domain_balance = {}
    for domain, values in domain_values.items():
        mean = sum(values) / len(values)
        per_domain_means[domain] = mean
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            # Balance = 1 - normalized variance (high balance = low variance within domain)
            per_domain_balance[domain] = max(0.0, 1.0 - variance)
        else:
            per_domain_balance[domain] = 1.0

    # Dominant distortion: domain with largest deviation from 1.0
    dominant_distortion = ""
    dominant_direction = ""
    max_deviation = 0.0
    for domain, mean in per_domain_means.items():
        deviation = abs(mean - 1.0)
        if deviation > max_deviation:
            max_deviation = deviation
            dominant_distortion = domain
            dominant_direction = "outward" if mean > 1.0 else "inward"

    return ShapeSnapshot(
        axes=sorted_axes,
        area_ratio=area_ratio,
        circularity_index=circularity,
        dominant_distortion=dominant_distortion,
        dominant_direction=dominant_direction,
        per_domain_means=per_domain_means,
        per_domain_balance=per_domain_balance,
        vertex_coordinates=vertices,
    )


def compute_delta(
    shape_a: ShapeSnapshot,
    shape_b: ShapeSnapshot,
    label_a: str = "fluid",
    label_b: str = "patterned",
) -> "ShapeDelta":
    """Compute the difference between two shapes."""
    from .contracts import ShapeDelta

    per_axis_delta: dict[str, float] = {}
    a_by_key = {ax.metric_key: ax.coupled for ax in shape_a.axes}
    b_by_key = {ax.metric_key: ax.coupled for ax in shape_b.axes}

    all_keys = set(a_by_key.keys()) | set(b_by_key.keys())
    for key in all_keys:
        val_a = a_by_key.get(key, 1.0)
        val_b = b_by_key.get(key, 1.0)
        per_axis_delta[key] = val_a - val_b

    # Domains with significant shift (>0.1 in mean)
    shifting = []
    for domain in set(shape_a.per_domain_means.keys()) | set(shape_b.per_domain_means.keys()):
        mean_a = shape_a.per_domain_means.get(domain, 1.0)
        mean_b = shape_b.per_domain_means.get(domain, 1.0)
        if abs(mean_a - mean_b) > 0.1:
            shifting.append(domain)

    area_delta = shape_a.area_ratio - shape_b.area_ratio
    circ_delta = shape_a.circularity_index - shape_b.circularity_index

    # Interpretation
    if label_a == "fluid" and label_b == "patterned":
        if abs(area_delta) < 0.05 and not shifting:
            interpretation = "Today is consistent with recent pattern."
        elif area_delta < -0.1:
            interpretation = f"Today is notably contracted vs recent pattern. {', '.join(shifting)} shifted."
        elif area_delta > 0.1:
            interpretation = f"Today is expanded vs recent pattern. {', '.join(shifting)} stronger."
        else:
            interpretation = f"Mild shift in {', '.join(shifting)}." if shifting else "Minor fluctuation."
    elif label_a == "patterned" and label_b == "static":
        if abs(area_delta) < 0.05 and not shifting:
            interpretation = "Recent pattern matches long-term baseline."
        elif area_delta < -0.1:
            interpretation = f"Trajectory is contracting. {', '.join(shifting)} declining from baseline."
        elif area_delta > 0.1:
            interpretation = f"Trajectory is expanding. {', '.join(shifting)} growing beyond baseline."
        else:
            interpretation = f"Gradual shift in {', '.join(shifting)}." if shifting else "Stable trajectory."
    else:
        interpretation = f"Delta between {label_a} and {label_b}."

    return ShapeDelta(
        layer_a=label_a,
        layer_b=label_b,
        area_delta=area_delta,
        circularity_delta=circ_delta,
        per_axis_delta=per_axis_delta,
        shifting_domains=shifting,
        interpretation=interpretation,
    )


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0


def _polygon_perimeter(vertices: list[tuple[float, float]]) -> float:
    """Perimeter of a polygon."""
    n = len(vertices)
    if n < 2:
        return 0.0
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        dx = vertices[j][0] - vertices[i][0]
        dy = vertices[j][1] - vertices[i][1]
        perimeter += math.sqrt(dx * dx + dy * dy)
    return perimeter
