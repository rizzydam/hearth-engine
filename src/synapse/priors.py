"""
Research priors -- hand-crafted synapses grounded in domain research
and behavioral science.

Three tiers:
1. CORE SYNAPSES: High-confidence, specific metric-to-metric connections
   with researched weights, lag windows, and threshold conditions.
   These are the backbone -- intra-domain, cascade chain, and the
   most validated cross-domain connections.

2. CATALOG SYNAPSES: Generated from metric catalog cross-domain pairs,
   but ONLY for connections not already covered by core synapses,
   and with lower weight (background signal, not primary).

3. DISCOVERED PRIORS: Learned from observation data -- relationships
   the system discovers through Hebbian co-movement detection.

A good synapse is SPECIFIC: exact metrics, directional weight, meaningful
lag window, threshold condition. A bad synapse is generic: "anything in
domain A affects anything in domain B."
"""

from __future__ import annotations

import logging

from .contracts import Synapse, _new_id
from .network import SynapseNetwork

log = logging.getLogger(__name__)

DECAY_RATES = {
    "research_prior": 0.005,
    "observed": 0.015,
    "ai_discovered": 0.025,
}


# =====================================================================
# TIER 1: CORE SYNAPSES -- high-confidence, specific, researched
# =====================================================================
# Format: (metric_a, metric_b, weight, direction, relationship,
#           lag_min, lag_max, threshold_a, research_basis)

INTRA_DOMAIN_SYNAPSES: list[tuple] = [
    # -- Fitness cluster --
    ("fitness.sleep_hours", "fitness.resting_heart_rate", 0.5, "positive", "reinforces",
     1, 7, None, "Adequate sleep lowers resting heart rate. Recovery effect."),
    ("fitness.exercise_minutes", "fitness.resting_heart_rate", 0.4, "positive", "reinforces",
     7, 30, None, "Regular exercise lowers resting heart rate over weeks. Cardio adaptation."),
    ("fitness.exercise_minutes", "fitness.sleep_hours", 0.35, "positive", "reinforces",
     0, 1, None, "Exercise improves sleep quality and duration. Same-day effect."),
    ("fitness.hydration_glasses", "fitness.exercise_minutes", 0.25, "positive", "reinforces",
     0, 1, None, "Proper hydration supports exercise performance."),

    # -- Project cluster --
    ("project.daily_focus_hours", "project.task_completion_rate", 0.6, "positive", "reinforces",
     0, 2, None, "Focus hours directly drive task completion. Execution requires deep work."),
    ("project.task_completion_rate", "project.backlog_count", 0.5, "negative", "reinforces",
     1, 7, None, "Higher completion rate reduces backlog over time. Throughput clears the queue."),
    ("project.review_turnaround_days", "project.team_velocity", 0.4, "negative", "threatens",
     1, 7, 5.0, "Slow reviews (>5 days) block team velocity. Bottleneck effect."),
    ("project.team_velocity", "project.backlog_count", 0.45, "negative", "reinforces",
     1, 14, None, "Higher velocity burns down backlog. Sustainable throughput."),

    # -- Business cluster --
    ("business.client_satisfaction", "business.monthly_revenue", 0.4, "positive", "reinforces",
     7, 30, None, "Happy clients lead to renewals and referrals. Revenue follows satisfaction."),
    ("business.pipeline_value", "business.monthly_revenue", 0.5, "positive", "reinforces",
     14, 60, None, "Strong pipeline converts to revenue over weeks. Leading indicator."),
    ("business.profit_margin", "business.pipeline_value", 0.3, "positive", "reinforces",
     7, 30, None, "Healthy margins enable reinvestment in sales. Margin funds growth."),
    ("business.invoice_days_outstanding", "business.profit_margin", 0.35, "negative", "threatens",
     7, 30, 45.0, "Slow collections (>45 days) strain cash flow and margin."),
]


CASCADE_CHAIN_SYNAPSES: list[tuple] = [
    # -- The primary cascade: fitness -> project -> business --

    # Sleep -> Focus (Walker 2017: sleep deprivation -> cognitive impairment, 1-2 day lag)
    ("fitness.sleep_hours", "project.daily_focus_hours", 0.65, "positive", "threatens",
     1, 2, None, "Walker (2017): Sleep <6h -> next-day cognitive impairment. Focus requires rest."),
    ("fitness.sleep_hours", "project.task_completion_rate", 0.5, "positive", "threatens",
     1, 3, None, "Sleep deprivation reduces task completion. Executive function degrades."),

    # Exercise -> Focus (acute cognitive benefits, 0-1 day lag)
    ("fitness.exercise_minutes", "project.daily_focus_hours", 0.4, "positive", "reinforces",
     0, 1, None, "Exercise improves cognitive function and focus. Acute effect."),

    # Project -> Business (work output -> business outcomes)
    ("project.task_completion_rate", "business.client_satisfaction", 0.45, "positive", "reinforces",
     3, 14, None, "Consistent delivery drives client satisfaction. Reliability earns trust."),
    ("project.team_velocity", "business.monthly_revenue", 0.35, "positive", "reinforces",
     7, 30, None, "Higher velocity means more deliverables. Output drives revenue."),

    # Business -> Fitness (stress feedback loop)
    ("business.profit_margin", "fitness.sleep_hours", 0.3, "positive", "threatens",
     1, 7, None, "Financial stress from low margins disrupts sleep. Worry keeps you up."),
]


CROSS_DOMAIN_SPECIFIC_SYNAPSES: list[tuple] = [
    # -- Fitness -> Business (indirect through cognitive function) --
    ("fitness.sleep_hours", "business.client_satisfaction", 0.3, "positive", "threatens",
     1, 7, None, "Sleep deprivation impairs interpersonal skills. Client interactions suffer."),
    ("fitness.resting_heart_rate", "project.daily_focus_hours", 0.25, "negative", "threatens",
     0, 3, 85.0, "Elevated RHR (>85) signals stress. Stress impairs focus."),

    # -- Business -> Project (resource constraints) --
    ("business.monthly_revenue", "project.team_velocity", 0.3, "positive", "reinforces",
     7, 30, None, "Revenue enables investment in tooling and team. Money buys capacity."),
    ("business.pipeline_value", "project.backlog_count", 0.35, "positive", "threatens",
     7, 30, None, "Growing pipeline adds to project backlog. Sales creates work."),

    # -- Fitness -> Business (the compound effect) --
    ("fitness.exercise_minutes", "business.profit_margin", 0.2, "positive", "reinforces",
     7, 30, None, "Regular exercise improves cognitive performance, indirectly supporting "
     "better business decisions and efficiency."),
]


BEHAVIORAL_SYNAPSES: list[tuple] = [
    # -- The burnout cascade --
    ("project.backlog_count", "fitness.sleep_hours", 0.35, "negative", "threatens",
     1, 7, 30.0, "Backlog above 30 creates stress that disrupts sleep. Cognitive weight."),
    ("project.daily_focus_hours", "fitness.exercise_minutes", 0.3, "negative", "threatens",
     0, 1, None, "Long focus days leave less time/energy for exercise. Sedentary risk."),

    # -- The recovery accelerants --
    ("fitness.sleep_hours", "project.review_turnaround_days", 0.3, "positive", "reinforces",
     1, 3, None, "Good sleep enables faster review turnaround. Cognitive function is up."),
    ("fitness.exercise_minutes", "fitness.sleep_hours", 0.4, "positive", "reinforces",
     0, 1, None, "Exercise improves sleep quality. Not just hours -- actual restoration."),

    # -- The virtuous cycle --
    ("business.client_satisfaction", "project.team_velocity", 0.25, "positive", "reinforces",
     1, 7, None, "Happy clients provide clearer feedback. Less rework, more velocity."),
    ("project.task_completion_rate", "fitness.sleep_hours", 0.2, "positive", "reinforces",
     0, 1, None, "Productive days reduce end-of-day anxiety. Completion enables rest."),
]


# =====================================================================
# ALL CORE SYNAPSES combined
# =====================================================================

ALL_CORE_SYNAPSES = (
    INTRA_DOMAIN_SYNAPSES
    + CASCADE_CHAIN_SYNAPSES
    + CROSS_DOMAIN_SPECIFIC_SYNAPSES
    + BEHAVIORAL_SYNAPSES
)


def seed_core_synapses(network: SynapseNetwork) -> int:
    """Seed Tier 1: hand-crafted, researched synapses with specific parameters."""
    created = 0
    for entry in ALL_CORE_SYNAPSES:
        (metric_a, metric_b, weight, direction, relationship,
         lag_min, lag_max, threshold_a, research_basis) = entry

        if network.find_existing(metric_a, metric_b):
            continue

        syn = Synapse(
            synapse_id=_new_id("core"),
            metric_a=metric_a,
            metric_b=metric_b,
            weight=weight,
            direction=direction,
            relationship=relationship,
            lag_window=(lag_min, lag_max),
            threshold_a=threshold_a,
            source="research_prior",
            research_basis=research_basis,
            confidence=0.7,  # Higher than catalog-generated
            decay_rate=DECAY_RATES["research_prior"],
        )
        network.add_synapse(syn)
        created += 1

    return created


def seed_catalog_background(network: SynapseNetwork, max_total: int = 200) -> int:
    """Seed Tier 2: catalog-generated synapses for background signal.

    Only creates synapses for metric pairs NOT already covered by core
    synapses. Lower weight (0.15-0.25) -- these are hypotheses, not claims.
    """
    try:
        from ..metric_catalog.catalog import get_cross_domain_pairs, ALL_METRICS
    except ImportError:
        return 0

    pairs = get_cross_domain_pairs()
    created = 0

    for metric_a, metric_b, relationship in pairs:
        if created >= max_total:
            break

        if network.find_existing(metric_a, metric_b):
            continue

        spec_a = ALL_METRICS.get(metric_a)
        spec_b = ALL_METRICS.get(metric_b)
        if not spec_a or not spec_b:
            continue

        # Lower weight for catalog-generated
        weight = 0.15 if relationship == "threatens" else 0.12

        direction = {"threatens": "negative", "protects": "positive",
                     "reinforces": "positive"}.get(relationship, "complex")

        syn = Synapse(
            synapse_id=_new_id("cat"),
            metric_a=metric_a,
            metric_b=metric_b,
            weight=weight,
            direction=direction,
            relationship=relationship,
            lag_window=(1, 7),
            source="research_prior",
            research_basis=f"Catalog: {spec_a.name} {relationship} {spec_b.name}",
            confidence=0.3,  # Low -- needs data confirmation
            decay_rate=DECAY_RATES["research_prior"],
        )
        network.add_synapse(syn)
        created += 1

    return created


def seed_from_catalog(network: SynapseNetwork, max_per_relationship: int = 300) -> int:
    """Full seeding: core synapses first, then catalog background.

    This is the entry point called by SynapseEngine on first boot.
    """
    core_count = seed_core_synapses(network)
    catalog_count = seed_catalog_background(network, max_total=max_per_relationship)

    total = core_count + catalog_count
    log.info(
        "Seeded synapse network: %d core + %d catalog = %d total",
        core_count, catalog_count, total,
    )
    return total
