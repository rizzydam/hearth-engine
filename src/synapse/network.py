"""
SynapseNetwork — the weighted directed graph of metric connections.

Nodes are metric keys. Edges are synapses. Indexes keep everything O(1).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .contracts import Synapse
from .store import SynapseStore


class SynapseNetwork:
    """The synapse graph with efficient lookup indexes."""

    def __init__(self, store: SynapseStore):
        self._store = store
        self._synapses: dict[str, Synapse] = {}
        self._by_metric: dict[str, list[str]] = defaultdict(list)
        self._by_domain_pair: dict[tuple[str, str], list[str]] = defaultdict(list)

    def load(self) -> None:
        """Load all synapses from the store and build indexes."""
        self._synapses.clear()
        self._by_metric.clear()
        self._by_domain_pair.clear()
        for syn in self._store.load_synapses():
            self._index(syn)

    def save(self) -> None:
        """Persist all synapses to the store."""
        self._store.save_synapses(list(self._synapses.values()))

    def _index(self, syn: Synapse) -> None:
        self._synapses[syn.synapse_id] = syn
        self._by_metric[syn.metric_a].append(syn.synapse_id)
        self._by_metric[syn.metric_b].append(syn.synapse_id)
        domain_a = _domain_from_key(syn.metric_a)
        domain_b = _domain_from_key(syn.metric_b)
        self._by_domain_pair[(domain_a, domain_b)].append(syn.synapse_id)
        if domain_a != domain_b:
            self._by_domain_pair[(domain_b, domain_a)].append(syn.synapse_id)

    def _unindex(self, syn: Synapse) -> None:
        sid = syn.synapse_id
        for key in (syn.metric_a, syn.metric_b):
            lst = self._by_metric.get(key, [])
            if sid in lst:
                lst.remove(sid)
        domain_a = _domain_from_key(syn.metric_a)
        domain_b = _domain_from_key(syn.metric_b)
        for pair in [(domain_a, domain_b), (domain_b, domain_a)]:
            lst = self._by_domain_pair.get(pair, [])
            if sid in lst:
                lst.remove(sid)

    # ── Mutation ──────────────────────────────────────────────────────

    def add_synapse(self, syn: Synapse) -> None:
        if syn.synapse_id in self._synapses:
            self._unindex(self._synapses[syn.synapse_id])
        self._index(syn)

    def remove_synapse(self, synapse_id: str) -> None:
        syn = self._synapses.pop(synapse_id, None)
        if syn:
            self._unindex(syn)

    # ── Queries ───────────────────────────────────────────────────────

    def get(self, synapse_id: str) -> Synapse | None:
        return self._synapses.get(synapse_id)

    def all(self) -> list[Synapse]:
        return list(self._synapses.values())

    def count(self) -> int:
        return len(self._synapses)

    def get_outgoing(self, metric_key: str) -> list[Synapse]:
        """Synapses where metric_key is the source (metric_a)."""
        return [self._synapses[sid] for sid in self._by_metric.get(metric_key, [])
                if self._synapses.get(sid) and self._synapses[sid].metric_a == metric_key]

    def get_incoming(self, metric_key: str) -> list[Synapse]:
        """Synapses where metric_key is the target (metric_b)."""
        return [self._synapses[sid] for sid in self._by_metric.get(metric_key, [])
                if self._synapses.get(sid) and self._synapses[sid].metric_b == metric_key]

    def get_connected(self, metric_key: str) -> list[Synapse]:
        """All synapses involving this metric (either direction)."""
        return [self._synapses[sid] for sid in self._by_metric.get(metric_key, [])
                if sid in self._synapses]

    def get_between_domains(self, domain_a: str, domain_b: str) -> list[Synapse]:
        return [self._synapses[sid] for sid in self._by_domain_pair.get((domain_a, domain_b), [])
                if sid in self._synapses]

    def get_by_relationship(self, relationship: str) -> list[Synapse]:
        return [s for s in self._synapses.values() if s.relationship == relationship]

    def domain_entanglement(self, domain_a: str, domain_b: str) -> float:
        synapses = self.get_between_domains(domain_a, domain_b)
        return sum(s.weight for s in synapses)

    def entanglement_matrix(self) -> dict[str, dict[str, float]]:
        domains = sorted(set(
            _domain_from_key(s.metric_a)
            for s in self._synapses.values()
        ) | set(
            _domain_from_key(s.metric_b)
            for s in self._synapses.values()
        ))
        matrix: dict[str, dict[str, float]] = {}
        for d1 in domains:
            matrix[d1] = {}
            for d2 in domains:
                matrix[d1][d2] = self.domain_entanglement(d1, d2)
        return matrix

    def all_metrics(self) -> set[str]:
        metrics = set()
        for s in self._synapses.values():
            metrics.add(s.metric_a)
            metrics.add(s.metric_b)
        return metrics

    def active_synapses(self, current_values: dict[str, float]) -> list[Synapse]:
        """Return synapses whose threshold conditions are met by current values."""
        result = []
        for syn in self._synapses.values():
            val_a = current_values.get(syn.metric_a)
            if val_a is None:
                continue
            if syn.threshold_a is not None and val_a < syn.threshold_a:
                continue
            val_b = current_values.get(syn.metric_b)
            if val_b is not None and syn.threshold_b is not None and val_b < syn.threshold_b:
                continue
            result.append(syn)
        return result

    def strongest_paths(self, metric_key: str, depth: int = 3) -> list[list[Synapse]]:
        """BFS following highest-weight outgoing edges up to depth."""
        paths: list[list[Synapse]] = []
        self._walk(metric_key, [], set(), depth, paths)
        paths.sort(key=lambda p: sum(s.weight for s in p) / len(p), reverse=True)
        return paths[:10]

    def _walk(self, metric: str, current_path: list[Synapse],
              visited: set[str], remaining: int,
              results: list[list[Synapse]]) -> None:
        if remaining <= 0:
            if current_path:
                results.append(list(current_path))
            return
        outgoing = self.get_outgoing(metric)
        outgoing.sort(key=lambda s: s.weight, reverse=True)
        for syn in outgoing[:5]:  # Limit branching
            if syn.synapse_id in visited:
                continue
            visited.add(syn.synapse_id)
            current_path.append(syn)
            results.append(list(current_path))
            self._walk(syn.metric_b, current_path, visited, remaining - 1, results)
            current_path.pop()
            visited.discard(syn.synapse_id)

    def find_existing(self, metric_a: str, metric_b: str) -> Synapse | None:
        """Find an existing synapse between two specific metrics."""
        for sid in self._by_metric.get(metric_a, []):
            syn = self._synapses.get(sid)
            if syn and syn.metric_a == metric_a and syn.metric_b == metric_b:
                return syn
            if syn and syn.metric_a == metric_b and syn.metric_b == metric_a:
                return syn
        return None


def _domain_from_key(metric_key: str) -> str:
    """Extract domain from metric key: 'fitness.sleep_hours' -> 'fitness'."""
    try:
        from ..metric_catalog.catalog import ALL_SENSES, ALL_METRICS
        spec = ALL_METRICS.get(metric_key)
        if spec:
            return ALL_SENSES[spec.sense].domain
    except ImportError:
        pass
    # Fallback: strip 'sense' suffix from sense name
    sense = metric_key.split(".")[0] if "." in metric_key else metric_key
    return sense.replace("sense", "")
