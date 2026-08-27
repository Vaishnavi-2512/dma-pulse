"""Two-stage AOA/RFO-style hyperparameter search utilities."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence
import numpy as np

Objective = Callable[[Mapping[str, float]], float]

@dataclass(frozen=True)
class SearchParameter:
    name: str
    low: float
    high: float
    integer: bool = False

@dataclass
class SearchResult:
    params: dict[str, float]
    score: float

def _decode(vector, space):
    result = {}
    for value, spec in zip(vector, space):
        clipped = float(np.clip(value, spec.low, spec.high))
        result[spec.name] = int(round(clipped)) if spec.integer else clipped
    return result

class AOAOptimizer:
    """Global exploration stage for the DMA-PULSE search pipeline."""
    def __init__(self, space, population=8, seed=42):
        self.space = list(space); self.population = population; self.rng = np.random.default_rng(seed)
    def optimize(self, objective, iterations=10):
        lows = np.array([p.low for p in self.space], dtype=float); highs = np.array([p.high for p in self.space], dtype=float)
        pop = self.rng.uniform(lows, highs, size=(self.population, len(self.space))); best_vector = pop[0].copy(); best_score = float("inf")
        for iteration in range(iterations):
            progress = (iteration + 1) / max(iterations, 1)
            for vector in pop:
                score = float(objective(_decode(vector, self.space)))
                if score < best_score: best_score, best_vector = score, vector.copy()
            scale = 1.0 - progress; noise = self.rng.normal(0.0, 1.0, size=pop.shape)
            attraction = self.rng.uniform(0.0, 1.0, size=pop.shape) * (best_vector - pop)
            pop = np.clip(pop + scale * attraction + 0.05 * scale * noise * (highs - lows), lows, highs)
        return SearchResult(_decode(best_vector, self.space), best_score)

class RFOOptimizer:
    """Local refinement stage seeded by the AOA search result."""
    def __init__(self, space, population=6, seed=123):
        self.space = list(space); self.population = population; self.rng = np.random.default_rng(seed)
    def optimize(self, objective, seed_result, iterations=8, radius=0.15):
        lows = np.array([p.low for p in self.space], dtype=float); highs = np.array([p.high for p in self.space], dtype=float); span = highs - lows
        center = np.array([seed_result.params.get(p.name, (p.low + p.high) / 2.0) for p in self.space], dtype=float)
        best_vector = center.copy(); best_score = seed_result.score
        for iteration in range(iterations):
            local_radius = radius * (1.0 - iteration / max(iterations, 1))
            candidates = center + self.rng.normal(0.0, local_radius, size=(self.population, len(self.space))) * span
            candidates = np.clip(candidates, lows, highs)
            for vector in candidates:
                score = float(objective(_decode(vector, self.space)))
                if score < best_score: best_score, best_vector, center = score, vector.copy(), vector.copy()
        return SearchResult(_decode(best_vector, self.space), best_score)

class DMAPulseOptimizer:
    """Convenience wrapper implementing the published AOA→RFO sequence."""
    def __init__(self, global_space, local_space, seed=42):
        self.global_optimizer = AOAOptimizer(global_space, seed=seed); self.local_optimizer = RFOOptimizer(local_space, seed=seed + 1)
    def optimize(self, objective):
        global_result = self.global_optimizer.optimize(objective); local_result = self.local_optimizer.optimize(objective, global_result)
        return global_result, local_result
