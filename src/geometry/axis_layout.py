"""
Axis layout — arranges metrics on a radial chart grouped by domain.

Each domain gets a sector of the circle (360/N degrees). Within each
sector, metrics are evenly spaced. Domains are ordered by entanglement
so that highly-connected domains are adjacent — making cross-domain
distortions visually contiguous.
"""

from __future__ import annotations

import math
from typing import Any

# Cache the layout since it doesn't change unless the catalog changes
_CACHED_LAYOUT: dict[str, float] | None = None
_CACHED_DOMAIN_ORDER: list[str] | None = None


def get_axis_angles() -> dict[str, float]:
    """Return metric_key -> angle_radians for all catalog metrics.

    Domains are arranged so the most entangled pairs are adjacent.
    Within each domain, metrics are evenly spaced.
    """
    global _CACHED_LAYOUT
    if _CACHED_LAYOUT is not None:
        return _CACHED_LAYOUT

    _CACHED_LAYOUT = _compute_layout()
    return _CACHED_LAYOUT


def get_domain_order() -> list[str]:
    """Return the ordered list of domains for the radial chart."""
    global _CACHED_DOMAIN_ORDER
    if _CACHED_DOMAIN_ORDER is not None:
        return _CACHED_DOMAIN_ORDER

    _CACHED_DOMAIN_ORDER = _compute_domain_order()
    return _CACHED_DOMAIN_ORDER


def get_domain_for_metric(metric_key: str) -> str:
    """Look up the domain for a metric key."""
    try:
        from ..metric_catalog.catalog import ALL_METRICS, ALL_SENSES
        spec = ALL_METRICS.get(metric_key)
        if spec:
            return ALL_SENSES[spec.sense].domain
    except ImportError:
        pass
    sense = metric_key.split(".")[0] if "." in metric_key else metric_key
    return sense.replace("sense", "")


def _compute_domain_order() -> list[str]:
    """Order domains so the most entangled pairs are adjacent.

    Uses a greedy nearest-neighbor heuristic on the entanglement matrix.
    """
    try:
        from ..metric_catalog.catalog import ALL_SENSES
        from ..synapse.network import SynapseNetwork
        from ..synapse.store import SynapseStore
        import tempfile
        from pathlib import Path

        domains = sorted(set(s.domain for s in ALL_SENSES.values()))

        # Try to get entanglement from a live network; fall back to default order
        try:
            store = SynapseStore()
            network = SynapseNetwork(store)
            network.load()
            if network.count() > 0:
                matrix = network.entanglement_matrix()
                return _greedy_order(domains, matrix)
        except Exception:
            pass

        # Default order: mental at center (most connected), alternating
        return _default_order(domains)

    except ImportError:
        return ["fitness", "project", "business"]


def _default_order(domains: list[str]) -> list[str]:
    """Default domain ordering with mental as the hub."""
    preferred = ["fitness", "project", "business"]
    return [d for d in preferred if d in domains] + [d for d in domains if d not in preferred]


def _greedy_order(domains: list[str], matrix: dict[str, dict[str, float]]) -> list[str]:
    """Greedy nearest-neighbor: start from most connected, add most entangled neighbor."""
    if not domains:
        return []

    # Start from domain with highest total entanglement
    totals = {}
    for d in domains:
        totals[d] = sum(matrix.get(d, {}).get(d2, 0) for d2 in domains if d2 != d)

    ordered = [max(totals, key=totals.get)]
    remaining = set(domains) - {ordered[0]}

    while remaining:
        last = ordered[-1]
        # Find most entangled neighbor not yet placed
        best = max(remaining, key=lambda d: matrix.get(last, {}).get(d, 0))
        ordered.append(best)
        remaining.remove(best)

    return ordered


def _compute_layout() -> dict[str, float]:
    """Compute angle_radians for each metric."""
    try:
        from ..metric_catalog.catalog import DOMAIN_METRICS
    except ImportError:
        return {}

    domain_order = get_domain_order()
    n_domains = max(len(domain_order), 1)
    sector_size = 2 * math.pi / n_domains

    layout: dict[str, float] = {}

    for i, domain in enumerate(domain_order):
        sector_start = i * sector_size
        metrics = DOMAIN_METRICS.get(domain, [])
        n_metrics = max(len(metrics), 1)

        for j, metric_key in enumerate(sorted(metrics)):
            # Evenly space within sector, with padding at edges
            offset = (j + 0.5) / n_metrics * sector_size
            layout[metric_key] = sector_start + offset

    return layout
