"""
Personal thresholds — configurable targets for every metric.

Stores per-metric targets and acceptable ranges in a YAML file at
``~/.home/thresholds.yaml``.  If the file doesn't exist, creates a
sensible-defaults version that is clearly marked for personalisation.

Also supports threshold *discovery* — inferring a reasonable target
from historical data when no explicit threshold is configured.

Migration note (Layer 6): path changed from ~/.hearth/thresholds.yaml
to ~/.home/thresholds.yaml.  Falls back to old path if it exists.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .contracts import ThresholdConfig, TimestampedValue


# We use a minimal YAML-like serialiser so we don't add a PyYAML dependency.
# The format is intentionally simple: one metric per section.

_DEFAULT_THRESHOLDS_YAML = """\
# =========================================================================
# Personal Thresholds — edit these to match YOUR life
# =========================================================================
# Each metric has:
#   target           — what you're aiming for
#   acceptable_low   — the floor of your "okay" range
#   acceptable_high  — the ceiling of your "okay" range
#   source           — "default" means these are starting points;
#                       change to "personal" once you've tuned them.
#
# The TrendEngine uses these to compute gap analysis and urgency.
# If a metric isn't listed here, the engine falls back to your personal
# baseline (your historical average).
# =========================================================================

sleep.duration:
  target: 7.5
  acceptable_low: 6.5
  acceptable_high: 9.0
  source: default

sleep.quality:
  target: 80
  acceptable_low: 60
  acceptable_high: 100
  source: default

spending.daily_total:
  target: 50
  acceptable_low: 0
  acceptable_high: 100
  source: default

exercise.minutes:
  target: 30
  acceptable_low: 15
  acceptable_high: 90
  source: default

mood.score:
  target: 7
  acceptable_low: 5
  acceptable_high: 10
  source: default

hydration.glasses:
  target: 8
  acceptable_low: 5
  acceptable_high: 12
  source: default
"""


class PersonalThresholds:
    """Load, query, and update personal metric thresholds."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            new_path = Path.home() / ".home" / "thresholds.yaml"
            old_path = Path.home() / ".hearth" / "thresholds.yaml"
            # Backward compat: use old path if it exists and new does not
            if old_path.exists() and not new_path.exists():
                path = old_path
            else:
                path = new_path
        self._path = path
        self._data: dict[str, ThresholdConfig] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, metric_key: str) -> ThresholdConfig | None:
        """Return the threshold config for a metric, or None."""
        return self._data.get(metric_key)

    def get_or_discover(
        self,
        metric_key: str,
        history: list[TimestampedValue] | None = None,
    ) -> ThresholdConfig | None:
        """Return explicit threshold, or discover one from history.

        Discovery logic:
            target = median of history
            acceptable_range = (p25, p75)
            source = "discovered"
        """
        existing = self.get(metric_key)
        if existing is not None:
            return existing

        if not history or len(history) < 5:
            return None

        vals = sorted(v.value for v in history)
        med = statistics.median(vals)
        p25 = _percentile(vals, 0.25)
        p75 = _percentile(vals, 0.75)

        return ThresholdConfig(
            target=round(med, 2),
            acceptable_range=(round(p25, 2), round(p75, 2)),
            source="discovered",
        )

    def update(
        self,
        metric_key: str,
        target: float,
        acceptable_range: tuple[float, float],
        source: str = "personal",
    ) -> None:
        """Set or update a threshold and persist to disk."""
        self._data[metric_key] = ThresholdConfig(
            target=target,
            acceptable_range=acceptable_range,
            source=source,
        )
        self._save()

    def all_keys(self) -> list[str]:
        """Return all configured metric keys."""
        return list(self._data.keys())

    # ------------------------------------------------------------------
    # Persistence (simple YAML-like format)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load thresholds from disk, creating defaults if needed."""
        if not self._path.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(_DEFAULT_THRESHOLDS_YAML, encoding="utf-8")

        text = self._path.read_text(encoding="utf-8")
        self._data = _parse_thresholds(text)

    def _save(self) -> None:
        """Write current thresholds back to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(_serialize_thresholds(self._data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Simple YAML-like parser / serialiser
# ---------------------------------------------------------------------------

def _parse_thresholds(text: str) -> dict[str, ThresholdConfig]:
    """Parse the simple threshold YAML format."""
    result: dict[str, ThresholdConfig] = {}
    current_key: str | None = None
    current: dict[str, Any] = {}

    for line in text.splitlines():
        stripped = line.strip()
        # Skip comments and blanks
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key (no leading whitespace, ends with colon)
        if not line.startswith(" ") and not line.startswith("\t") and stripped.endswith(":"):
            # Flush previous
            if current_key and current:
                result[current_key] = _dict_to_config(current)
            current_key = stripped.rstrip(":")
            current = {}
        elif current_key and ":" in stripped:
            # Sub-key
            k, v = stripped.split(":", 1)
            k = k.strip()
            v = v.strip()
            current[k] = v

    # Flush last
    if current_key and current:
        result[current_key] = _dict_to_config(current)

    return result


def _dict_to_config(d: dict[str, Any]) -> ThresholdConfig:
    """Convert parsed dict to ThresholdConfig."""
    target = float(d.get("target", 0))
    low = float(d.get("acceptable_low", target * 0.8))
    high = float(d.get("acceptable_high", target * 1.2))
    source = str(d.get("source", "default"))
    return ThresholdConfig(
        target=target,
        acceptable_range=(low, high),
        source=source,
    )


def _serialize_thresholds(data: dict[str, ThresholdConfig]) -> str:
    """Serialize thresholds back to the simple YAML format."""
    lines = [
        "# Personal Thresholds",
        "# Edit these to match YOUR life",
        "",
    ]
    for key, cfg in sorted(data.items()):
        lines.append(f"{key}:")
        lines.append(f"  target: {cfg.target}")
        lines.append(f"  acceptable_low: {cfg.acceptable_range[0]}")
        lines.append(f"  acceptable_high: {cfg.acceptable_range[1]}")
        lines.append(f"  source: {cfg.source}")
        lines.append("")
    return "\n".join(lines)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile from a sorted list."""
    import math
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    k = pct * (n - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_values[lo]
    frac = k - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac
