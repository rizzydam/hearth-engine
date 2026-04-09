"""
Hard rules -- non-negotiable safety thresholds.

These are the floor. Synapses provide context and nuance. Hard rules
provide certainty. When a hard rule fires, it always wins.
"""

from __future__ import annotations

from .contracts import HardRule, HardRuleFire


# The rules themselves -- these don't learn, decay, or negotiate.
HARD_RULES: list[HardRule] = [
    HardRule(
        rule_id="cognitive_impairment",
        metric_key="fitness.sleep_hours",
        threshold=5.0,
        comparison="lt",
        consecutive_days=3,
        severity="critical",
        message="Sleep below 5 hours for 3+ days. Cognitive impairment is clinically certain. "
                "Decision quality is degraded -- defer major decisions if possible.",
        clinical_basis="Walker (2017): <6h sleep for 5+ nights produces measurable cognitive decline "
                       "equivalent to legal intoxication.",
    ),
    HardRule(
        rule_id="elevated_heart_rate",
        metric_key="fitness.resting_heart_rate",
        threshold=100.0,
        comparison="gt",
        consecutive_days=3,
        severity="critical",
        message="Resting heart rate above 100 BPM for 3+ days. Medical attention may be warranted.",
        clinical_basis="Sustained resting tachycardia above 100 BPM indicates physiological stress.",
    ),
    HardRule(
        rule_id="project_stall",
        metric_key="project.task_completion_rate",
        threshold=0.2,
        comparison="lt",
        consecutive_days=7,
        severity="warning",
        message="Task completion below 20% for a week. Execution capacity may be compromised.",
        clinical_basis="Sustained low completion signals burnout, scope overload, or tooling failure.",
    ),
    HardRule(
        rule_id="focus_collapse",
        metric_key="project.daily_focus_hours",
        threshold=1.0,
        comparison="lt",
        consecutive_days=5,
        severity="warning",
        message="Less than 1 hour of focus per day for 5+ days. Deep work capacity is gone.",
        clinical_basis="Sustained inability to focus signals burnout or environmental/organizational dysfunction.",
    ),
    HardRule(
        rule_id="revenue_crisis",
        metric_key="business.monthly_revenue",
        threshold=1000.0,
        comparison="lt",
        consecutive_days=1,
        severity="critical",
        message="Monthly revenue below $1,000. Business sustainability is at risk.",
        clinical_basis="Revenue below operating expenses signals unsustainable business model.",
    ),
    HardRule(
        rule_id="margin_collapse",
        metric_key="business.profit_margin",
        threshold=0.05,
        comparison="lt",
        consecutive_days=1,
        severity="warning",
        message="Profit margin below 5%. Operating at near-loss.",
        clinical_basis="Sub-5% margin leaves no buffer for unexpected costs.",
    ),
]

# Map rule -> domain for corroboration
_RULE_DOMAINS: dict[str, str] = {
    "cognitive_impairment": "fitness",
    "elevated_heart_rate": "fitness",
    "project_stall": "project",
    "focus_collapse": "project",
    "revenue_crisis": "business",
    "margin_collapse": "business",
}


class HardRuleEngine:
    """Evaluate hard rules against current metric values.

    Hard rules are simple threshold checks. They don't need baselines,
    trends, or synapses. If the condition is true, it fires.
    """

    def __init__(self, rules: list[HardRule] | None = None):
        self._rules = rules or HARD_RULES

    def evaluate(self, current_values: dict[str, float]) -> list[HardRuleFire]:
        """Check all rules against current values. Returns fired rules.

        Note: consecutive_days checking requires historical data. For now,
        we fire on single-point violations and note the consecutive
        requirement in the message. The learning engine will add temporal
        tracking as observation history grows.
        """
        fires: list[HardRuleFire] = []
        for rule in self._rules:
            value = current_values.get(rule.metric_key)
            if value is None:
                continue
            if rule.check(value):
                fires.append(HardRuleFire(
                    rule_id=rule.rule_id,
                    severity=rule.severity,
                    message=rule.message,
                    metric_key=rule.metric_key,
                    current_value=value,
                ))
        return fires

    @staticmethod
    def rule_to_domain(rule_id: str) -> str:
        return _RULE_DOMAINS.get(rule_id, "other")

    @staticmethod
    def corroborate(
        synapse_severities: dict[str, float],
        hard_fires: list[HardRuleFire],
    ) -> dict[str, float]:
        """Merge hard rule fires with synapse-derived domain severities.

        For each domain, final severity = MAX(hard_rule, synapse).
        Hard rules always win when they fire.
        """
        result = dict(synapse_severities)
        for fire in hard_fires:
            domain = _RULE_DOMAINS.get(fire.rule_id, "other")
            hard_sev = 1.0 if fire.severity == "critical" else 0.7
            result[domain] = max(result.get(domain, 0.0), hard_sev)
        return result
