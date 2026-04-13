# Hearth Engine

A reusable intelligence pipeline for multi-domain metric tracking, cross-metric relationship discovery, geometric state assessment, and mode-based triage.

## Architecture

Hearth Engine implements a compound intelligence pipeline:

```
Raw Metrics → Trend Analysis → Synapse Learning → Geometric Assessment → Mode-Based Triage
```

### Core Components

| Component | Purpose |
|-----------|---------|
| **Metric Catalog** | Registry of typed metrics with bilateral thresholds (critical/warning/healthy/thriving), cross-domain relationship hints, and evaluation logic |
| **Synapse Network** | Weighted directed graph of metric-to-metric relationships with Hebbian learning (reinforce confirmed connections, decay unconfirmed) |
| **Assessment Engine** | Reads current metric values against the synapse network to produce activations, cascade predictions, domain severities, resilience scores |
| **Geometry Layer** | Normalizes metrics, arranges on radial axes, applies force propagation through synapses, computes shape metrics (area = health, circularity = balance) |
| **Trend Engine** | Computes baselines, detects trends (direction, velocity), measures gaps from healthy ranges, scores urgency |
| **Mode Detector** | Derives system mode (stable/drifting/sliding/crisis) from geometric assessment with hysteresis for stability |
| **Vision (Adversarial)** | Runs recommendations through skeptical personas before presenting them |
| **Lint Engine** | Periodic health-check of the intelligence system itself — stale synapses, untested hypotheses, data gaps |

### Key Design Principles

1. **Zero external dependencies** — All computation uses Python stdlib only (`math`, `statistics`, `dataclasses`)
2. **Bilateral thresholds** — Every metric has critical/warning/healthy/thriving ranges, not just good/bad
3. **Hebbian learning** — Connections strengthen when confirmed by data, weaken when contradicted, decay without reinforcement
4. **Geometry as assessment** — Multi-dimensional state represented as a shape, not a scorecard
5. **Mode-based cognitive load limiting** — Crisis mode shows 1 task, stable mode shows 6
6. **Adversarial validation** — AI recommendations pass through skeptical review before reaching the user

## Usage

```python
from hearth_engine.metric_catalog import MetricCatalog, MetricSpec
from hearth_engine.synapse import SynapseNetwork, HebbianLearner, AssessmentEngine
from hearth_engine.geometry import ProfileGeometry
from hearth_engine.trend import TrendEngine

# 1. Define your metrics
catalog = MetricCatalog()
catalog.register(MetricSpec(
    key="health.sleep_hours",
    name="Sleep Hours",
    unit="hours",
    healthy_min=7.0, healthy_max=9.0,
    warning_threshold=6.0, critical_threshold=5.0,
    thriving_threshold=8.0,
))

# 2. Build synapse network (connections between metrics)
network = SynapseNetwork()
learner = HebbianLearner(network)

# 3. Feed observations — the system learns relationships
learner.observe_and_discover(metric_observations)

# 4. Assess current state
engine = AssessmentEngine(network, catalog)
assessment = engine.assess(current_values)

# 5. Compute geometry (optional)
geometry = ProfileGeometry(catalog, network)
shape = geometry.assess(current_values)

# 6. Derive mode
mode = shape.mode  # "stable", "drifting", "sliding", "crisis"
```

## Directory Structure

```
src/
    metric_catalog/     # MetricSpec, MetricCatalog, evaluation engine
    synapse/            # Network, Hebbian learning, assessment, cascades, hard rules
    geometry/           # Normalization, axis layout, force propagation, shape metrics
    trend/              # Baselines, trend detection, gap analysis, urgency scoring
    triage/             # Mode detection, task limiting
    vision/             # Adversarial persona validation
    lint/               # System health checking
    pipeline/           # Aggregation and orchestration patterns
examples/
    fitness_tracker/    # Apply to fitness domain
    project_manager/    # Apply to project management
tests/
```

## Standalone Value

| Component | Reusability |
|-----------|------------|
| Metric Catalog | Any app tracking metrics with thresholds |
| Hebbian Synapse Learning | Any domain with multiple correlated time-series |
| Geometric Assessment | Any multi-dimensional health/state visualization |
| Mode-Based Triage | Any task/recommendation system needing cognitive load management |
| Adversarial Validation | Any AI recommendation system needing skepticism |

## License

MIT — see [LICENSE](LICENSE)
