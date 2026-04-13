"""
Metric Catalog -- the single source of truth for what every domain exports.

This module defines every metric in the ecosystem. It serves as:
    1. The contract each domain implements against
    2. The registry TrendEngine computes from
    3. The vocabulary the synapse layer connects through
    4. The coverage map that tells us what we're blind to

Each metric has a canonical key in `domain.metric_name` format,
healthy ranges derived from research, and metadata the synapse layer
needs to build connections.

Design: metrics are grouped by domain. Each DomainContract lists the
metrics that domain MUST export, with enough metadata to compute
baselines, detect drift, grade health, and feed synapses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MetricUnit(str, Enum):
    """Standard units for metric values."""
    HOURS       = "hours"
    SCORE_1_10  = "score_1_10"
    SCORE_0_100 = "score_0_100"
    SCORE_0_1   = "score_0_1"
    COUNT       = "count"
    DOLLARS     = "dollars"
    PERCENT     = "percent"
    BPM         = "bpm"
    POUNDS      = "pounds"
    DAYS        = "days"
    RATIO       = "ratio"
    BOOLEAN     = "boolean"
    MINUTES     = "minutes"
    GLASSES     = "glasses"


class DataSource(str, Enum):
    """How this metric gets its data."""
    COLLECTION  = "collection"   # From a LocalStore/SqliteStore collection
    SUMMARY     = "summary"      # Derived from domain summary/export
    COMPUTED    = "computed"      # Computed from other metrics at export time
    EXTERNAL    = "external"     # From external API


@dataclass(frozen=True)
class MetricSpec:
    """Complete specification for a single metric.

    This is the atomic unit of the metric catalog. Everything the system
    needs to know about a metric -- how to read it, what's normal, when
    to worry, AND when to celebrate.

    The system tracks both directions: degradation triggers alarms,
    but sustained health triggers recognition. A system that only
    criticizes becomes another source of dread. One that also notices
    wins becomes something worth believing in.
    """
    # Identity
    key: str                            # "fitness.sleep_hours"
    sense: str                          # "fitness"
    name: str                           # Human-readable: "Sleep Hours"
    description: str                    # What this measures and why it matters

    # Data source
    source: DataSource = DataSource.COLLECTION
    collection: str = ""                # Store collection name
    store_field: str = ""               # Field within collection record
    aggregation: str = "mean"           # "mean" | "sum" | "last" | "max" | "min"

    # Value semantics
    unit: MetricUnit = MetricUnit.SCORE_0_1
    value_range: tuple[float, float] = (0.0, 1.0)  # Possible min/max
    higher_is_better: bool = True

    # -- Negative thresholds (alarm side) ---------------------------------
    healthy_range: tuple[float, float] = (0.0, 1.0)
    warning_low: float | None = None    # Below this = warning
    warning_high: float | None = None   # Above this = warning
    critical_low: float | None = None   # Below this = critical
    critical_high: float | None = None  # Above this = critical

    # -- Positive thresholds (recognition side) ---------------------------
    # Thriving: when the value is not just healthy but *excellent*.
    # The system should notice this and say so.
    thriving_threshold: float | None = None  # Above/below this (per higher_is_better) = thriving
    thriving_streak_days: int = 3            # Consecutive days in thriving range to celebrate
    # Momentum: when the trend itself is the win, regardless of absolute value.
    # "You've improved 4 days in a row" matters even if you're still in warning range.
    momentum_matters: bool = True            # Whether improving trajectory is a positive signal
    # What the win looks like in human terms
    win_description: str = ""                # e.g., "Sleep is locked in. This is the foundation."
    streak_description: str = ""             # e.g., "X days of solid sleep. Routine is holding."
    momentum_description: str = ""           # e.g., "Sleep is coming back. Keep going."
    # Record tracking -- some metrics should celebrate all-time or period records
    track_personal_best: bool = False        # Whether to flag new records

    # -- Reinforcement (what good values enable) --------------------------
    # When this metric is thriving, which domains get a tailwind?
    # Distinct from 'protects' -- protects means "shields from harm",
    # reinforces means "actively makes things better when strong."
    reinforces: tuple[str, ...] = ()

    # Cadence and quality
    expected_cadence_hours: float = 24.0  # How often we expect new data
    data_quality_default: float = 0.6     # Default quality (0.6 = self-report)
    min_data_points: int = 7              # Minimum for baseline computation

    # Cross-domain relevance (which domains this metric protects or threatens)
    protects: tuple[str, ...] = ()        # Domains shielded from harm when healthy
    threatens: tuple[str, ...] = ()       # Domains affected when this degrades

    # Synapse hints
    cascade_position: int = 0             # 0=early trigger, 5=late consequence
    is_hard_rule_metric: bool = False     # Used in clinical hard rules

    def evaluate(self, value: float) -> str:
        """Evaluate a value against all thresholds. Returns a level string.

        Levels (ordered best to worst):
            thriving -> healthy -> warning -> critical

        This is used by both the alarm system and the recognition system.
        """
        # Check thriving first
        if self.thriving_threshold is not None:
            if self.higher_is_better and value >= self.thriving_threshold:
                return "thriving"
            if not self.higher_is_better and value <= self.thriving_threshold:
                return "thriving"

        # Critical
        if self.critical_low is not None and value <= self.critical_low:
            return "critical"
        if self.critical_high is not None and value >= self.critical_high:
            return "critical"

        # Warning
        if self.warning_low is not None and value <= self.warning_low:
            return "warning"
        if self.warning_high is not None and value >= self.warning_high:
            return "warning"

        # Healthy range
        low, high = self.healthy_range
        if low <= value <= high:
            return "healthy"

        return "warning"  # Outside healthy but below warning thresholds


@dataclass
class SenseContract:
    """The complete metric contract for one domain.

    Lists every metric the domain is expected to export, plus metadata
    about the domain itself.
    """
    sense_name: str
    display_name: str
    domain: str                          # "fitness", "project", "business"
    description: str
    metrics: dict[str, MetricSpec]       # key -> MetricSpec
    store_collections: tuple[str, ...]   # What data stores this domain manages

    def metric_keys(self) -> list[str]:
        return list(self.metrics.keys())

    def collection_backed(self) -> dict[str, MetricSpec]:
        return {k: v for k, v in self.metrics.items()
                if v.source == DataSource.COLLECTION}

    def summary_derived(self) -> dict[str, MetricSpec]:
        return {k: v for k, v in self.metrics.items()
                if v.source in (DataSource.SUMMARY, DataSource.COMPUTED)}


# =====================================================================
# FITNESS DOMAIN
# =====================================================================

FITNESS = SenseContract(
    sense_name="fitness",
    display_name="Fitness",
    domain="fitness",
    description="Tracks sleep, exercise, vitals, and physical health patterns.",
    store_collections=("vitals", "sleep_log", "exercise_log"),
    metrics={
        "fitness.sleep_hours": MetricSpec(
            key="fitness.sleep_hours",
            sense="fitness",
            name="Sleep Hours",
            description="Hours of sleep per night. The single most predictive metric for "
                        "next-day cognitive function, mood, and decision quality. "
                        "7-9 hours is the clinical target (Walker 2017).",
            source=DataSource.COLLECTION,
            collection="sleep_log",
            store_field="hours",
            aggregation="mean",
            unit=MetricUnit.HOURS,
            value_range=(0.0, 16.0),
            higher_is_better=True,
            healthy_range=(7.0, 9.0),
            warning_low=6.0,
            warning_high=10.0,
            critical_low=5.0,
            thriving_threshold=8.0,
            thriving_streak_days=5,
            win_description="Solid sleep. Tomorrow has a real chance.",
            streak_description="nights of 8+ hours. The foundation is holding.",
            momentum_description="Sleep is improving. Everything downstream gets easier.",
            reinforces=("project", "business"),
            expected_cadence_hours=24.0,
            data_quality_default=0.6,
            protects=("fitness", "project", "business"),
            threatens=("project", "business"),
            cascade_position=0,
            is_hard_rule_metric=True,
        ),
        "fitness.exercise_minutes": MetricSpec(
            key="fitness.exercise_minutes",
            sense="fitness",
            name="Daily Exercise Minutes",
            description="Minutes of exercise per day. WHO recommends 150+ min moderate or "
                        "75+ min vigorous per week. 30-60 min/day is the target.",
            source=DataSource.COLLECTION,
            collection="exercise_log",
            store_field="minutes",
            aggregation="sum",
            unit=MetricUnit.MINUTES,
            value_range=(0.0, 300.0),
            higher_is_better=True,
            healthy_range=(30.0, 60.0),
            warning_low=15.0,
            critical_low=0.0,
            thriving_threshold=45.0,
            thriving_streak_days=5,
            win_description="Solid exercise today. Consistency over intensity.",
            streak_description="days hitting exercise target. This is becoming a habit.",
            momentum_description="Exercise frequency is building. One more session than last week.",
            track_personal_best=True,
            reinforces=("project",),
            expected_cadence_hours=24.0,
            data_quality_default=0.7,
            protects=("fitness", "project"),
            threatens=("fitness", "project"),
            cascade_position=1,
        ),
        "fitness.resting_heart_rate": MetricSpec(
            key="fitness.resting_heart_rate",
            sense="fitness",
            name="Resting Heart Rate",
            description="Resting BPM. 60-80 normal. Elevated RHR correlates with stress, "
                        "poor fitness, or illness. Trend matters more than absolute.",
            source=DataSource.COLLECTION,
            collection="vitals",
            store_field="heart_rate",
            aggregation="mean",
            unit=MetricUnit.BPM,
            value_range=(30.0, 200.0),
            higher_is_better=False,
            healthy_range=(60.0, 80.0),
            warning_high=85.0,
            critical_high=100.0,
            warning_low=45.0,
            thriving_threshold=55.0,
            win_description="Resting heart rate is excellent. Fitness is showing.",
            momentum_description="RHR is trending down. Cardiovascular health improving.",
            track_personal_best=True,
            reinforces=("fitness",),
            expected_cadence_hours=24.0,
            data_quality_default=0.9,
            protects=("fitness",),
            threatens=("fitness",),
            cascade_position=1,
        ),
        "fitness.hydration_glasses": MetricSpec(
            key="fitness.hydration_glasses",
            sense="fitness",
            name="Hydration (Glasses)",
            description="Glasses of water per day. 6-10 is the target range. "
                        "Dehydration impairs cognition and energy.",
            source=DataSource.COLLECTION,
            collection="vitals",
            store_field="water_glasses",
            aggregation="sum",
            unit=MetricUnit.GLASSES,
            value_range=(0.0, 20.0),
            higher_is_better=True,
            healthy_range=(6.0, 10.0),
            warning_low=4.0,
            critical_low=2.0,
            thriving_threshold=8.0,
            thriving_streak_days=5,
            win_description="Well hydrated today. Cognition and energy benefit.",
            streak_description="days of good hydration. The habit is holding.",
            momentum_description="Hydration is improving. Small wins add up.",
            reinforces=("fitness", "project"),
            expected_cadence_hours=24.0,
            data_quality_default=0.6,
            protects=("fitness",),
            threatens=("fitness",),
            cascade_position=2,
        ),
        "fitness.body_weight_lbs": MetricSpec(
            key="fitness.body_weight_lbs",
            sense="fitness",
            name="Body Weight",
            description="Body weight in pounds. Trend matters more than absolute value. "
                        "Rapid changes (5+ lbs/week) signal stress eating or illness.",
            source=DataSource.COLLECTION,
            collection="vitals",
            store_field="weight",
            aggregation="last",
            unit=MetricUnit.POUNDS,
            value_range=(50.0, 500.0),
            higher_is_better=False,
            healthy_range=(150.0, 180.0),
            warning_high=190.0,
            critical_high=210.0,
            thriving_threshold=165.0,
            momentum_matters=True,
            momentum_description="Weight is trending toward target. Steady wins.",
            expected_cadence_hours=168.0,
            data_quality_default=0.9,
            protects=("fitness",),
            threatens=("fitness",),
            cascade_position=2,
        ),
    },
)


# =====================================================================
# PROJECT MANAGEMENT DOMAIN
# =====================================================================

PROJECT = SenseContract(
    sense_name="project",
    display_name="Project",
    domain="project",
    description="Tracks task completion, focus time, backlog, reviews, and team velocity.",
    store_collections=("tasks", "sprints", "reviews", "focus_sessions"),
    metrics={
        "project.task_completion_rate": MetricSpec(
            key="project.task_completion_rate",
            sense="project",
            name="Task Completion Rate",
            description="Tasks completed / total assigned. Core work performance metric. "
                        "Declining rate signals capacity overload or disengagement.",
            source=DataSource.COMPUTED,
            unit=MetricUnit.RATIO,
            value_range=(0.0, 1.0),
            higher_is_better=True,
            healthy_range=(0.6, 0.9),
            warning_low=0.4,
            critical_low=0.2,
            thriving_threshold=0.85,
            thriving_streak_days=5,
            win_description="Getting things done. Execution is strong.",
            streak_description="weeks of high completion. Reliable and productive.",
            momentum_description="Completion rate is climbing. Momentum building.",
            reinforces=("business",),
            expected_cadence_hours=168.0,
            data_quality_default=0.7,
            protects=("project", "business"),
            threatens=("project", "business"),
            cascade_position=2,
        ),
        "project.daily_focus_hours": MetricSpec(
            key="project.daily_focus_hours",
            sense="project",
            name="Daily Focus Hours",
            description="Hours of uninterrupted deep work per day. "
                        "4-8 hours is the productive range. Below 2 signals "
                        "context-switching or meeting overload.",
            source=DataSource.COLLECTION,
            collection="focus_sessions",
            store_field="hours",
            aggregation="sum",
            unit=MetricUnit.HOURS,
            value_range=(0.0, 16.0),
            higher_is_better=True,
            healthy_range=(4.0, 8.0),
            warning_low=2.0,
            critical_low=1.0,
            thriving_threshold=6.0,
            thriving_streak_days=5,
            win_description="6+ hours of deep work. Serious output today.",
            streak_description="days of sustained focus. Deep work is a habit.",
            momentum_description="Focus hours are increasing. Protecting the time.",
            track_personal_best=True,
            reinforces=("project", "business"),
            expected_cadence_hours=24.0,
            data_quality_default=0.7,
            protects=("project",),
            threatens=("project", "business"),
            cascade_position=1,
        ),
        "project.backlog_count": MetricSpec(
            key="project.backlog_count",
            sense="project",
            name="Backlog Count",
            description="Total items in the backlog. 5-20 is manageable. "
                        "Above 30 signals prioritization failure or scope creep.",
            source=DataSource.SUMMARY,
            unit=MetricUnit.COUNT,
            value_range=(0.0, 200.0),
            higher_is_better=False,
            healthy_range=(5.0, 20.0),
            warning_high=30.0,
            critical_high=50.0,
            thriving_threshold=10.0,
            win_description="Backlog is lean. Only real work in the queue.",
            momentum_description="Backlog is shrinking. Getting ahead of the work.",
            reinforces=("project",),
            expected_cadence_hours=168.0,
            data_quality_default=0.8,
            protects=(),
            threatens=("project",),
            cascade_position=3,
        ),
        "project.review_turnaround_days": MetricSpec(
            key="project.review_turnaround_days",
            sense="project",
            name="Review Turnaround (Days)",
            description="Average days from review request to completion. "
                        "1-3 days is responsive. Above 5 blocks the team.",
            source=DataSource.COMPUTED,
            unit=MetricUnit.DAYS,
            value_range=(0.0, 30.0),
            higher_is_better=False,
            healthy_range=(1.0, 3.0),
            warning_high=5.0,
            critical_high=7.0,
            thriving_threshold=1.0,
            win_description="Reviews are fast. Nothing blocked waiting on feedback.",
            momentum_description="Review time is getting faster. Team throughput up.",
            reinforces=("project",),
            expected_cadence_hours=168.0,
            data_quality_default=0.8,
            protects=("project",),
            threatens=("project",),
            cascade_position=3,
        ),
        "project.team_velocity": MetricSpec(
            key="project.team_velocity",
            sense="project",
            name="Team Velocity",
            description="Story points or tasks completed per sprint. "
                        "15-30 is typical for a small team. Declining velocity "
                        "signals tech debt, burnout, or planning problems.",
            source=DataSource.SUMMARY,
            unit=MetricUnit.COUNT,
            value_range=(0.0, 100.0),
            higher_is_better=True,
            healthy_range=(15.0, 30.0),
            warning_low=10.0,
            critical_low=5.0,
            thriving_threshold=25.0,
            thriving_streak_days=3,
            win_description="Velocity is strong. Team is in flow.",
            streak_description="sprints of high velocity. Predictable delivery.",
            momentum_description="Velocity is climbing. Improvements are landing.",
            track_personal_best=True,
            reinforces=("project", "business"),
            expected_cadence_hours=336.0,  # Bi-weekly sprints
            data_quality_default=0.8,
            protects=("project", "business"),
            threatens=("project", "business"),
            cascade_position=2,
        ),
    },
)


# =====================================================================
# BUSINESS DOMAIN
# =====================================================================

BUSINESS = SenseContract(
    sense_name="business",
    display_name="Business",
    domain="business",
    description="Tracks revenue, client satisfaction, pipeline, invoicing, and profitability.",
    store_collections=("clients", "revenue", "invoices", "leads", "expenses"),
    metrics={
        "business.monthly_revenue": MetricSpec(
            key="business.monthly_revenue",
            sense="business",
            name="Monthly Revenue",
            description="Total revenue for the current month. The primary top-line metric. "
                        "Below $3,000/month signals sustainability risk.",
            source=DataSource.SUMMARY,
            unit=MetricUnit.DOLLARS,
            value_range=(0.0, 1000000.0),
            higher_is_better=True,
            healthy_range=(5000.0, 15000.0),
            warning_low=3000.0,
            critical_low=1000.0,
            thriving_threshold=12000.0,
            win_description="Revenue is strong this month. Business is sustaining.",
            momentum_description="Revenue is trending up. Growth is real.",
            track_personal_best=True,
            reinforces=("business", "project"),
            expected_cadence_hours=720.0,
            data_quality_default=0.9,
            protects=("business",),
            threatens=("business", "project"),
            cascade_position=1,
        ),
        "business.client_satisfaction": MetricSpec(
            key="business.client_satisfaction",
            sense="business",
            name="Client Satisfaction",
            description="Average client satisfaction score 1-10. Below 5 signals "
                        "service quality problems. Retention risk below 7.",
            source=DataSource.SUMMARY,
            unit=MetricUnit.SCORE_1_10,
            value_range=(1.0, 10.0),
            higher_is_better=True,
            healthy_range=(7.0, 10.0),
            warning_low=5.0,
            critical_low=3.0,
            thriving_threshold=9.0,
            thriving_streak_days=7,
            win_description="Clients are happy. Retention is solid.",
            streak_description="weeks of high satisfaction. Reputation is building.",
            momentum_description="Client satisfaction improving. Service quality showing.",
            reinforces=("business",),
            expected_cadence_hours=168.0,
            data_quality_default=0.7,
            protects=("business",),
            threatens=("business",),
            cascade_position=2,
        ),
        "business.pipeline_value": MetricSpec(
            key="business.pipeline_value",
            sense="business",
            name="Pipeline Value",
            description="Total value of open/prospect leads. Zero pipeline = future revenue "
                        "risk. Healthy is 3x monthly revenue.",
            source=DataSource.SUMMARY,
            unit=MetricUnit.DOLLARS,
            value_range=(0.0, 1000000.0),
            higher_is_better=True,
            healthy_range=(10000.0, 50000.0),
            warning_low=5000.0,
            critical_low=1000.0,
            thriving_threshold=30000.0,
            win_description="Strong pipeline. Future revenue is building.",
            track_personal_best=True,
            reinforces=("business",),
            expected_cadence_hours=168.0,
            data_quality_default=0.8,
            protects=("business",),
            threatens=("business",),
            cascade_position=2,
        ),
        "business.invoice_days_outstanding": MetricSpec(
            key="business.invoice_days_outstanding",
            sense="business",
            name="Invoice Days Outstanding",
            description="Average days from invoice sent to payment received. Above 45 days "
                        "signals cash flow risk.",
            source=DataSource.COMPUTED,
            unit=MetricUnit.DAYS,
            value_range=(0.0, 180.0),
            higher_is_better=False,
            healthy_range=(15.0, 30.0),
            warning_high=45.0,
            critical_high=60.0,
            thriving_threshold=15.0,
            win_description="Getting paid fast. Cash flow is healthy.",
            momentum_description="Collection time is shrinking. Clients are paying quicker.",
            reinforces=("business",),
            expected_cadence_hours=720.0,
            data_quality_default=0.9,
            protects=(),
            threatens=("business",),
            cascade_position=3,
        ),
        "business.profit_margin": MetricSpec(
            key="business.profit_margin",
            sense="business",
            name="Profit Margin",
            description="Monthly (revenue - expenses) / revenue. Below 10% is thin. "
                        "Below 5% is crisis. Negative = losing money.",
            source=DataSource.COMPUTED,
            unit=MetricUnit.RATIO,
            value_range=(-1.0, 1.0),
            higher_is_better=True,
            healthy_range=(0.2, 0.4),
            warning_low=0.1,
            critical_low=0.05,
            thriving_threshold=0.35,
            win_description="35%+ margin. Business is healthy and profitable.",
            momentum_description="Margin is improving. Efficiency gains showing.",
            track_personal_best=True,
            reinforces=("business", "project"),
            expected_cadence_hours=720.0,
            data_quality_default=0.9,
            protects=("business",),
            threatens=("business", "project"),
            cascade_position=2,
        ),
    },
)


# =====================================================================
# FULL CATALOG
# =====================================================================

ALL_SENSES: dict[str, SenseContract] = {
    "fitness": FITNESS,
    "project": PROJECT,
    "business": BUSINESS,
}

# Flat lookup: metric_key -> MetricSpec
ALL_METRICS: dict[str, MetricSpec] = {}
for _contract in ALL_SENSES.values():
    ALL_METRICS.update(_contract.metrics)

# Every metric that participates in hard rules
HARD_RULE_METRICS: dict[str, MetricSpec] = {
    k: v for k, v in ALL_METRICS.items() if v.is_hard_rule_metric
}

# Domain -> list of metric keys
DOMAIN_METRICS: dict[str, list[str]] = {}
for _key, _spec in ALL_METRICS.items():
    _domain = ALL_SENSES[_spec.sense].domain
    DOMAIN_METRICS.setdefault(_domain, []).append(_key)


# =====================================================================
# BACKWARD COMPATIBILITY -- TrendEngine MetricDefinition generation
# =====================================================================

def to_trend_registry() -> dict[str, "MetricDefinition"]:
    """Convert collection-backed MetricSpecs into TrendEngine MetricDefinitions.

    This bridges the metric catalog to the existing TrendEngine, allowing
    TrendEngine to be driven from the catalog instead of its hardcoded registry.
    Only collection-backed metrics are included -- summary-derived metrics are
    handled separately by TrendEngine._extract_summary_metrics().
    """
    from .trend.contracts import MetricDefinition

    registry: dict[str, MetricDefinition] = {}
    for spec in ALL_METRICS.values():
        if spec.source != DataSource.COLLECTION:
            continue
        if not spec.collection or not spec.store_field:
            continue

        registry[spec.key] = MetricDefinition(
            key=spec.key,
            collection=spec.collection,
            field=spec.store_field,
            higher_is_better=spec.higher_is_better,
            aggregation=spec.aggregation,
            sense_name=spec.sense,
        )

    return registry


# =====================================================================
# CROSS-DOMAIN CONNECTION MAP
# =====================================================================
# Pre-computed from protects/threatens fields. This tells the synapse
# layer which metric pairs are worth monitoring for co-movement.

def get_cross_domain_pairs() -> list[tuple[str, str, str]]:
    """Return (metric_a, metric_b, relationship_type) for all cross-domain pairs.

    Three relationship types:
    - "threatens": when A degrades, B is at risk
    - "protects": when A is strong, B is shielded
    - "reinforces": when A is thriving, B gets a tailwind

    These are research-prior candidates for synapse connections.
    """
    pairs: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for key_a, spec_a in ALL_METRICS.items():
        domain_a = ALL_SENSES[spec_a.sense].domain

        # Threatening connections: when A degrades, B is at risk
        for threatened_domain in spec_a.threatens:
            for key_b in DOMAIN_METRICS.get(threatened_domain, []):
                if key_a == key_b:
                    continue
                triple = (min(key_a, key_b), max(key_a, key_b), "threatens")
                if triple not in seen:
                    seen.add(triple)
                    pairs.append((key_a, key_b, "threatens"))

        # Protective connections: when A is strong, B is protected
        for protected_domain in spec_a.protects:
            if protected_domain == domain_a:
                continue
            for key_b in DOMAIN_METRICS.get(protected_domain, []):
                if key_a == key_b:
                    continue
                triple = (min(key_a, key_b), max(key_a, key_b), "protects")
                if triple not in seen:
                    seen.add(triple)
                    pairs.append((key_a, key_b, "protects"))

        # Reinforcement connections: when A is thriving, B gets a tailwind
        for reinforced_domain in spec_a.reinforces:
            for key_b in DOMAIN_METRICS.get(reinforced_domain, []):
                if key_a == key_b:
                    continue
                triple = (min(key_a, key_b), max(key_a, key_b), "reinforces")
                if triple not in seen:
                    seen.add(triple)
                    pairs.append((key_a, key_b, "reinforces"))

    return pairs


# Pre-computed positive metrics: those with thriving thresholds defined
THRIVING_METRICS: dict[str, MetricSpec] = {
    k: v for k, v in ALL_METRICS.items() if v.thriving_threshold is not None
}

# Metrics that track streaks
STREAK_METRICS: dict[str, MetricSpec] = {
    k: v for k, v in ALL_METRICS.items()
    if v.thriving_threshold is not None and v.streak_description
}

# Metrics that celebrate momentum (improving trend)
MOMENTUM_METRICS: dict[str, MetricSpec] = {
    k: v for k, v in ALL_METRICS.items()
    if v.momentum_matters and v.momentum_description
}

# Metrics that track records
PERSONAL_BEST_METRICS: dict[str, MetricSpec] = {
    k: v for k, v in ALL_METRICS.items() if v.track_personal_best
}


def catalog_stats() -> dict[str, int | float]:
    """Summary statistics for the metric catalog."""
    total = len(ALL_METRICS)
    collection_backed = sum(1 for m in ALL_METRICS.values() if m.source == DataSource.COLLECTION)
    summary_derived = sum(1 for m in ALL_METRICS.values() if m.source == DataSource.SUMMARY)
    computed = sum(1 for m in ALL_METRICS.values() if m.source == DataSource.COMPUTED)
    hard_rules = len(HARD_RULE_METRICS)
    cross_pairs = get_cross_domain_pairs()
    thriving = len(THRIVING_METRICS)
    streaks = len(STREAK_METRICS)
    momentum = len(MOMENTUM_METRICS)
    personal_bests = len(PERSONAL_BEST_METRICS)

    # Count by relationship type
    threatens_count = sum(1 for _, _, r in cross_pairs if r == "threatens")
    protects_count = sum(1 for _, _, r in cross_pairs if r == "protects")
    reinforces_count = sum(1 for _, _, r in cross_pairs if r == "reinforces")

    return {
        "total_metrics": total,
        "collection_backed": collection_backed,
        "summary_derived": summary_derived,
        "computed": computed,
        "hard_rule_metrics": hard_rules,
        "cross_domain_pairs": len(cross_pairs),
        "threatens_pairs": threatens_count,
        "protects_pairs": protects_count,
        "reinforces_pairs": reinforces_count,
        "thriving_metrics": thriving,
        "streak_metrics": streaks,
        "momentum_metrics": momentum,
        "personal_best_metrics": personal_bests,
        "senses": len(ALL_SENSES),
        "domains": len(set(s.domain for s in ALL_SENSES.values())),
    }
