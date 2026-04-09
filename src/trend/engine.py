"""
TrendEngine — the orchestrator that ties everything together.

This is the single entry point for trend intelligence. It:
    1. Reads historical data for every registered metric
    2. Computes personal baselines
    3. Analyses trends (direction, velocity, change points)
    4. Calculates gaps from targets
    5. Scores urgency based on trajectory
    6. Packages everything into a TrendReport

The engine is designed to be called once per aggregation cycle. Its
output enriches the AggregatedState so downstream consumers (pattern
detection, recommender, briefings) can use trend-aware reasoning.

All computation is stdlib-only. No numpy, no scipy, no external deps.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .baselines import BaselineComputer
from .contracts import (
    MetricDefinition,
    MetricSnapshot,
    TimestampedValue,
    TrendDirection,
    TrendReport,
)
from .gaps import GapAnalyzer
from .staleness import StalenessCalculator
from .thresholds import PersonalThresholds
from .trends import TrendComputer
from .urgency import TrajectoryScorer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric registry — what the engine knows how to track
# ---------------------------------------------------------------------------
# Loaded from the metric catalog (metric_catalog.catalog), which is
# the single source of truth for all metrics across every domain.
# Legacy keys (sleep.hours, etc.) are aliased for backward compat.

def _build_metric_registry() -> dict[str, MetricDefinition]:
    """Build the metric registry from the metric catalog.

    TrendEngine uses legacy-format keys (sleep.hours, mood.score, etc.)
    for backward compatibility with tests and downstream consumers.
    The catalog's sensename.metric_name keys are used directly by the
    synapse layer — they don't go through TrendEngine.

    This function reads catalog MetricSpecs and maps them to the legacy
    key format that TrendEngine consumers expect.
    """
    # Legacy key -> catalog key mapping.
    # Note: stress.severity reads from stress_log (collection), not the
    # summary-derived compound stress_load. checkin.* fields map to the
    # catalog's checkins-backed metrics.
    _LEGACY_TO_CATALOG = {
        "sleep.hours": "healthsense.sleep_hours",
        "sleep.quality": "healthsense.sleep_quality",
        "mood.score": "mindsense.mood_score",
        "checkin.anxiety": "mindsense.anxiety_level",
        "checkin.energy": "mindsense.energy_level",
        "spending.daily": "financesense.spending_daily",
        "health.daily_steps": "healthsense.daily_steps",
        "health.exercise_frequency": "healthsense.exercise_sessions",
    }

    registry: dict[str, MetricDefinition] = {}

    try:
        from ..metric_catalog.catalog import ALL_METRICS
        catalog = ALL_METRICS
    except ImportError:
        catalog = {}

    # Build legacy-keyed definitions from catalog specs
    for legacy_key, catalog_key in _LEGACY_TO_CATALOG.items():
        spec = catalog.get(catalog_key)
        if spec is None or not spec.collection or not spec.store_field:
            continue
        registry[legacy_key] = MetricDefinition(
            key=legacy_key,
            collection=spec.collection,
            field=spec.store_field,
            higher_is_better=spec.higher_is_better,
            aggregation=spec.aggregation,
            sense_name=spec.sense,
        )

    # Metrics that don't map 1:1 to catalog entries (different collection/field)
    if "stress.severity" not in registry:
        registry["stress.severity"] = MetricDefinition(
            "stress.severity", "stress_log", "severity", False, "mean", "mindsense")
    if "checkin.mood" not in registry:
        registry["checkin.mood"] = MetricDefinition(
            "checkin.mood", "checkins", "mood", True, "mean", "mindsense")

    # Fallback: if catalog import failed, use hardcoded defaults
    if not registry:
        registry = {
            "sleep.hours": MetricDefinition("sleep.hours", "sleep_log", "hours", True, "mean", "healthsense"),
            "sleep.quality": MetricDefinition("sleep.quality", "sleep_log", "quality", True, "mean", "healthsense"),
            "mood.score": MetricDefinition("mood.score", "mood_log", "score", True, "mean", "mindsense"),
            "stress.severity": MetricDefinition("stress.severity", "stress_log", "severity", False, "mean", "mindsense"),
            "checkin.mood": MetricDefinition("checkin.mood", "checkins", "mood", True, "mean", "mindsense"),
            "checkin.anxiety": MetricDefinition("checkin.anxiety", "checkins", "anxiety", False, "mean", "mindsense"),
            "checkin.energy": MetricDefinition("checkin.energy", "checkins", "energy", True, "mean", "mindsense"),
            "spending.daily": MetricDefinition("spending.daily", "transactions", "amount", False, "sum", "financesense"),
            "health.daily_steps": MetricDefinition("health.daily_steps", "vitals", "steps", True, "sum", "healthsense"),
            "health.exercise_frequency": MetricDefinition("health.exercise_frequency", "vitals", "exercise_sessions", True, "sum", "healthsense"),
        }

    return registry


METRIC_REGISTRY: dict[str, MetricDefinition] = _build_metric_registry()


class TrendEngine:
    """Compute trend intelligence for all registered metrics.

    Parameters
    ----------
    store : object
        A LocalStore or SqliteStore instance used to load historical data.
        Must have a ``load(collection)`` method.
    thresholds_path : Path, optional
        Override path for the thresholds YAML file.
    half_life_days : float
        Staleness half-life for data weighting.
    baseline_window_days : int
        How far back to look for baseline computation.
    """

    def __init__(
        self,
        store: Any = None,
        thresholds_path: Any = None,
        half_life_days: float = 7.0,
        baseline_window_days: int = 90,
    ) -> None:
        self._store = store
        self._staleness = StalenessCalculator(half_life_days)
        self._baseline_computer = BaselineComputer(self._staleness)
        self._thresholds = PersonalThresholds(thresholds_path) if thresholds_path else None
        self._scorer = TrajectoryScorer()
        self._baseline_window = baseline_window_days

    def compute(
        self,
        state: Any = None,
        store: Any = None,
        now: datetime | None = None,
    ) -> TrendReport:
        """Run the full trend analysis pipeline for all registered metrics.

        Parameters
        ----------
        state : AggregatedState, optional
            Current aggregated state (used for sense-name mapping).
        store : object, optional
            Override the store for this computation.
        now : datetime, optional
            Reference time.

        Returns
        -------
        TrendReport
        """
        if now is None:
            now = datetime.now(timezone.utc)
        active_store = store or self._store

        report = TrendReport()
        declining_count = 0

        # First pass: compute trends to count declining metrics
        snapshots: dict[str, MetricSnapshot] = {}
        trend_results: dict[str, Any] = {}

        for key, metric_def in METRIC_REGISTRY.items():
            try:
                values = self._load_values(active_store, metric_def, now)
                if not values:
                    snapshots[key] = MetricSnapshot(
                        baseline=None, trend=None, gap=None,
                        urgency=None, staleness=0.0,
                    )
                    continue

                # Staleness of most recent value
                most_recent = max(values, key=lambda v: v.timestamp)
                freshness = self._staleness.freshness(most_recent.age_days(now))

                # Baseline
                baseline = self._baseline_computer.compute(
                    key, values, self._baseline_window, now,
                )

                # Trend
                trend_computer = TrendComputer(
                    self._staleness,
                    higher_is_better=metric_def.higher_is_better,
                )
                trend = trend_computer.analyze(key, values, baseline, now)

                if trend.direction == TrendDirection.DECLINING:
                    declining_count += 1

                trend_results[key] = {
                    "values": values,
                    "baseline": baseline,
                    "trend": trend,
                    "freshness": freshness,
                    "metric_def": metric_def,
                }

            except Exception as e:
                log.warning("TrendEngine: error computing %s: %s", key, e)
                snapshots[key] = MetricSnapshot(
                    baseline=None, trend=None, gap=None,
                    urgency=None, staleness=0.0,
                )

        # Second pass: score urgency with compound awareness
        for key, data in trend_results.items():
            try:
                baseline = data["baseline"]
                trend = data["trend"]
                freshness = data["freshness"]
                metric_def = data["metric_def"]
                values = data["values"]

                # Gap analysis
                gap = None
                if trend.current_value is not None:
                    threshold = None
                    if self._thresholds:
                        threshold = self._thresholds.get_or_discover(key, values)

                    gap_analyzer = GapAnalyzer(
                        higher_is_better=metric_def.higher_is_better,
                    )
                    gap = gap_analyzer.analyze(
                        key, trend.current_value, baseline, trend,
                        threshold, now,
                    )

                # Urgency (subtract 1 from declining_count for THIS metric if it's declining)
                other_declining = declining_count
                if trend.direction == TrendDirection.DECLINING:
                    other_declining = max(declining_count - 1, 0)

                urgency = self._scorer.score(
                    trend, gap, freshness, other_declining,
                )

                snapshots[key] = MetricSnapshot(
                    baseline=baseline,
                    trend=trend,
                    gap=gap,
                    urgency=urgency,
                    staleness=freshness,
                )

            except Exception as e:
                log.warning("TrendEngine: error scoring %s: %s", key, e)
                snapshots[key] = MetricSnapshot(
                    baseline=data.get("baseline"),
                    trend=data.get("trend"),
                    gap=None,
                    urgency=None,
                    staleness=data.get("freshness", 0.0),
                )

        report.metrics = snapshots
        report.overall_trajectory = self._summarize(snapshots)
        return report

    def enrich_state(self, state: Any, report: TrendReport) -> Any:
        """Attach a TrendReport to an AggregatedState.

        This is a non-destructive operation — it adds the trend_report
        attribute without modifying any existing state data.

        Parameters
        ----------
        state : AggregatedState
            The current state to enrich.
        report : TrendReport
            The computed trend report.

        Returns
        -------
        AggregatedState
            The same state object, now with ``trend_report`` attached.
        """
        state.trend_report = report
        return state

    # ------------------------------------------------------------------
    # Summary-derived metrics (not in store collections)
    # ------------------------------------------------------------------

    def _extract_summary_metrics(self, state: Any) -> dict[str, float]:
        """Read metrics that are derived from AggregatedState sense summaries.

        These metrics (credit_utilization, savings_rate, unread_email_ratio,
        education_engagement) live in sense summary fields rather than in
        store collections, so the normal collection-based pipeline cannot
        track them.  This method extracts their current values directly.

        Parameters
        ----------
        state : AggregatedState
            The current aggregated state containing sense exports.

        Returns
        -------
        dict mapping metric name to current float value.
        """
        values: dict[str, float] = {}

        if state is None:
            return values

        senses = getattr(state, "senses", {})

        # --- FinanceSense summary ---
        fs = senses.get("financesense")
        if fs:
            summary = getattr(fs, "summary", {})
            cu = summary.get("credit_utilization")
            if cu is not None:
                values["summary.credit_utilization"] = float(cu)
            sr = summary.get("savings_rate")
            if sr is not None:
                values["summary.savings_rate"] = float(sr)

        # --- CommSense summary ---
        cs = senses.get("commsense")
        if cs:
            summary = getattr(cs, "summary", {})
            total = max(int(summary.get("total_emails", 1)), 1)
            unread = int(summary.get("unread_count", 0))
            values["summary.unread_email_ratio"] = unread / total

        # --- CareerSense summary ---
        crs = senses.get("careersense")
        if crs:
            summary = getattr(crs, "summary", {})
            ee = summary.get("education_sessions_per_week")
            if ee is not None:
                values["summary.education_engagement"] = float(ee)

        return values

    # ------------------------------------------------------------------
    # Drift detection (replaces drift_detector.detect_drift)
    # ------------------------------------------------------------------

    # Domain mapping and directionality — built from metric catalog with
    # legacy fallbacks for summary-derived metrics.
    @staticmethod
    def _build_domain_map() -> dict[str, str]:
        try:
            from ..metric_catalog.catalog import ALL_METRICS, ALL_SENSES
            mapping = {}
            for key, spec in ALL_METRICS.items():
                mapping[key] = ALL_SENSES[spec.sense].domain
            # Legacy aliases
            mapping.update({
                "sleep.hours": "health", "sleep.quality": "health",
                "mood.score": "mental", "stress.severity": "mental",
                "checkin.mood": "mental", "checkin.anxiety": "mental",
                "checkin.energy": "mental", "spending.daily": "financial",
                "health.daily_steps": "health", "health.exercise_frequency": "health",
                "summary.credit_utilization": "financial",
                "summary.savings_rate": "financial",
                "summary.unread_email_ratio": "comm",
                "summary.education_engagement": "career",
            })
            return mapping
        except ImportError:
            return {
                "sleep.hours": "health", "sleep.quality": "health",
                "mood.score": "mental", "stress.severity": "mental",
                "checkin.mood": "mental", "checkin.anxiety": "mental",
                "checkin.energy": "mental", "spending.daily": "financial",
                "health.daily_steps": "health", "health.exercise_frequency": "health",
                "summary.credit_utilization": "financial",
                "summary.savings_rate": "financial",
                "summary.unread_email_ratio": "comm",
                "summary.education_engagement": "career",
            }

    @staticmethod
    def _build_higher_is_worse() -> dict[str, bool]:
        try:
            from ..metric_catalog.catalog import ALL_METRICS
            return {k: not v.higher_is_better for k, v in ALL_METRICS.items()
                    if not v.higher_is_better}
        except ImportError:
            return {
                "spending.daily": True, "stress.severity": True,
                "checkin.anxiety": True, "summary.credit_utilization": True,
                "summary.unread_email_ratio": True,
            }

    _METRIC_DOMAIN_MAP: dict[str, str] = {}
    _HIGHER_IS_WORSE: dict[str, bool] = {}
    _maps_initialized: bool = False

    def _ensure_maps(self) -> None:
        if not TrendEngine._maps_initialized:
            TrendEngine._METRIC_DOMAIN_MAP = TrendEngine._build_domain_map()
            TrendEngine._HIGHER_IS_WORSE = TrendEngine._build_higher_is_worse()
            TrendEngine._maps_initialized = True

    def get_drifts(self, report: TrendReport | None = None, state: Any = None) -> list[dict]:
        """Compute drift signals from TrendEngine data.

        A metric is drifting when its current value deviates more than
        1.5 standard deviations from its personal baseline.  This mirrors
        the logic from drift_detector.detect_drift() but reads directly
        from TrendEngine baselines and trend data.

        Parameters
        ----------
        report : TrendReport, optional
            A pre-computed TrendReport. If None, ``self.compute(state=state)``
            is called.
        state : AggregatedState, optional
            Used to compute the report (if not provided) and to extract
            summary-derived metrics.

        Returns
        -------
        list of dict in the same shape as DriftMetric.to_dict().
        """
        self._ensure_maps()

        if report is None:
            report = self.compute(state=state)

        drifts: list[dict] = []

        # --- Collection-backed metrics from the TrendReport ---
        for key, snap in report.metrics.items():
            if snap.baseline is None or snap.trend is None:
                continue
            if snap.trend.direction == TrendDirection.INSUFFICIENT_DATA:
                continue
            if snap.trend.current_value is None:
                continue

            baseline = snap.baseline
            current = snap.trend.current_value
            stddev = baseline.std_dev
            mean = baseline.mean

            if stddev == 0 and mean == 0:
                continue

            threshold = 1.5 * stddev if stddev > 0 else 0.0
            deviation = current - mean

            if abs(deviation) <= threshold:
                continue

            # Direction
            higher_is_worse = self._HIGHER_IS_WORSE.get(key, False)
            if higher_is_worse:
                direction = "declining" if deviation > 0 else "improving"
            else:
                direction = "declining" if deviation < 0 else "improving"

            # Severity
            if stddev > 0:
                abs_z = abs(deviation) / stddev
                if abs_z > 2.5:
                    severity = "critical"
                elif abs_z > 1.5:
                    severity = "warning"
                else:
                    severity = "info"
            else:
                severity = "warning"

            domain = self._METRIC_DOMAIN_MAP.get(key, "other")

            drifts.append({
                "domain": domain,
                "metric_name": key,
                "current_value": round(current, 4),
                "baseline_mean": round(mean, 4),
                "baseline_stddev": round(stddev, 4),
                "direction": direction,
                "days_drifting": snap.trend.duration_days,
                "severity": severity,
                "last_updated": snap.trend.trend_started.isoformat()
                if snap.trend.trend_started else "",
            })

        # --- Summary-derived metrics (simple deviation check) ---
        if state is not None:
            summary_vals = self._extract_summary_metrics(state)
            for skey, current in summary_vals.items():
                # Summary metrics don't have TrendEngine baselines, so we
                # can't do z-score drift.  They are included for completeness
                # and can be compared to fixed thresholds if needed.
                pass

        return drifts

    @staticmethod
    def count_declining_domains(drifts: list[dict]) -> int:
        """Count unique domains with at least one declining drift signal."""
        declining = set()
        for d in drifts:
            if d.get("direction") == "declining":
                declining.add(d.get("domain"))
        return len(declining)

    @staticmethod
    def domain_health_from_drifts(drifts: list[dict]) -> dict[str, float]:
        """Compute per-domain health scores from drift signals.

        Starts at 1.0 per domain. Each drift signal reduces it.
        Mirrors drift_detector.domain_health_from_drifts().
        """
        all_domains = [
            "financial", "health", "mental", "career",
            "comm", "social", "family", "routine",
        ]
        health: dict[str, float] = {d: 1.0 for d in all_domains}

        for d in drifts:
            domain = d.get("domain", "other")
            if domain not in health:
                health[domain] = 1.0

            if d.get("direction") == "declining":
                severity = d.get("severity", "info")
                penalty = {"info": 0.05, "warning": 0.15, "critical": 0.30}.get(
                    severity, 0.10,
                )
                days = min(d.get("days_drifting", 0), 5)
                penalty *= days / 3  # longer drift = bigger penalty
                health[domain] = max(0.0, health[domain] - penalty)
            elif d.get("direction") == "improving":
                bonus = 0.05
                health[domain] = min(1.0, health[domain] + bonus)

        return {k: round(v, 2) for k, v in health.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_values(
        self,
        store: Any,
        metric_def: MetricDefinition,
        now: datetime,
    ) -> list[TimestampedValue]:
        """Load raw values from the store and convert to TimestampedValue list.

        If ``metric_def.sense_name`` is set and no explicit store was passed
        to the engine, we create a LocalStore for that sense to read its
        historical data directly from ``~/.home/senses/<sense_name>/``.

        When ``metric_def.aggregation == "sum"``, records are grouped by
        date and summed BEFORE creating TimestampedValues. This fixes the
        spending metric which otherwise treats individual transactions as
        separate data points instead of daily totals.
        """
        active_store = store
        if active_store is None and metric_def.sense_name:
            try:
                from ..storage import LocalStore
                active_store = LocalStore(metric_def.sense_name)
            except Exception as e:
                log.debug("Could not create store for %s: %s", metric_def.sense_name, e)
                return []
        if active_store is None:
            return []

        try:
            records = active_store.load(metric_def.collection)
        except Exception as e:
            log.debug("Could not load collection %s: %s", metric_def.collection, e)
            return []

        # Parse all records into (timestamp, value) pairs
        parsed: list[tuple[datetime, float]] = []
        for record in records:
            try:
                raw_value = record.get(metric_def.field)
                if raw_value is None:
                    continue
                value = float(raw_value)

                # Try multiple timestamp field names
                ts_raw = (
                    record.get("timestamp")
                    or record.get("date")
                    or record.get("created_at")
                    or record.get("recorded_at")
                )
                if ts_raw is None:
                    continue

                if isinstance(ts_raw, datetime):
                    ts = ts_raw
                elif isinstance(ts_raw, str):
                    ts = datetime.fromisoformat(ts_raw)
                else:
                    continue

                # Ensure timezone-aware
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                parsed.append((ts, value))

            except (ValueError, TypeError, KeyError):
                continue

        # When aggregation is "sum", group by date and sum amounts.
        # This turns individual transactions into daily totals — essential
        # for spending metrics where 5 transactions of $20 should be $100/day,
        # not 5 separate data points.
        if metric_def.aggregation == "sum" and parsed:
            from collections import defaultdict
            daily_sums: dict[str, float] = defaultdict(float)
            daily_ts: dict[str, datetime] = {}
            for ts, value in parsed:
                date_key = ts.strftime("%Y-%m-%d")
                daily_sums[date_key] += value
                # Keep the latest timestamp for each date
                if date_key not in daily_ts or ts > daily_ts[date_key]:
                    daily_ts[date_key] = ts

            return [
                TimestampedValue(timestamp=daily_ts[dk], value=daily_sums[dk])
                for dk in sorted(daily_sums.keys())
            ]

        return [TimestampedValue(timestamp=ts, value=val) for ts, val in parsed]

    def _summarize(self, snapshots: dict[str, MetricSnapshot]) -> str:
        """Build a one-paragraph overall trajectory summary."""
        declining = []
        improving = []
        urgent = []

        for key, snap in snapshots.items():
            name = key.replace(".", " ").replace("_", " ")
            if snap.trend and snap.trend.direction == TrendDirection.DECLINING:
                declining.append(name)
            elif snap.trend and snap.trend.direction == TrendDirection.IMPROVING:
                improving.append(name)
            if snap.urgency and snap.urgency.level.value in ("act", "urgent"):
                urgent.append(name)

        parts: list[str] = []

        if urgent:
            parts.append(
                f"Needs attention: {', '.join(urgent)}."
            )
        if declining:
            parts.append(
                f"Declining: {', '.join(declining)}."
            )
        if improving:
            parts.append(
                f"Improving: {', '.join(improving)}."
            )

        total = len(snapshots)
        active = sum(
            1 for s in snapshots.values()
            if s.trend and s.trend.direction != TrendDirection.INSUFFICIENT_DATA
        )
        if total > 0:
            parts.append(f"Tracking {active}/{total} metrics with sufficient data.")

        return " ".join(parts) if parts else "No metrics with sufficient data for trend analysis."
