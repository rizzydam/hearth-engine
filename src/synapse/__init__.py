"""Synapse Network — Hebbian learning, assessment, and cascade prediction."""
from .contracts import Synapse, MetricObservation, SynapseAssessment, ActivatedSynapse
from .network import SynapseNetwork
from .learning import HebbianLearner
from .assessment import AssessmentEngine
