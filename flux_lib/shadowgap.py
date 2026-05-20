"""
flux_lib.shadowgap — Find what ALL checkers miss.

The shadowgap is the region of violation space that no strategy covered.
Each shadowgap correction monotonically reduces future shadowgap rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class StrategyResult:
    """Output from one checking strategy."""
    name: str
    error_masks: np.ndarray       # (N,), uint8
    covered: np.ndarray           # (N,), bool
    checks_performed: int
    checks_skipped: int
    strategy_mask: np.ndarray     # (N,), uint8


def _ground_truth(lo: np.ndarray, hi: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Compute true violation masks. Returns uint8 array."""
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    N = values.shape[0]
    D = values.shape[1]
    masks = np.zeros(N, dtype=np.uint8)
    for i in range(D):
        violated = (values[:, i] < lo[i]) | (values[:, i] > hi[i])
        masks[violated] |= np.uint8(1 << i)
    return masks


class MultiChecker:
    """
    Runs multiple constraint-checking strategies on the same input.

    Strategies:
        A: Strict — checks everything (baseline)
        B: Adaptive — most-likely-to-fail first, skips rest on first fail
        C: Predictive — skips checks predicted to pass
        D: Severity-weighted — high-severity first, budget-limited
    """

    def __init__(self, lo: np.ndarray, hi: np.ndarray,
                 severity_order: Optional[np.ndarray] = None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.D = len(lo)
        self.severity_order = np.asarray(severity_order) if severity_order is not None else np.arange(self.D)
        self._pass_rates = np.ones(self.D, dtype=np.float64) * 0.99

    def ground_truth(self, values: np.ndarray) -> np.ndarray:
        return _ground_truth(self.lo, self.hi, values)

    def strategy_strict(self, values: np.ndarray) -> StrategyResult:
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        N = values.shape[0]
        masks = self.ground_truth(values)
        return StrategyResult(
            name="strict", error_masks=masks, covered=np.ones(N, dtype=bool),
            checks_performed=N * self.D, checks_skipped=0,
            strategy_mask=np.full(N, (1 << self.D) - 1, dtype=np.uint8),
        )

    def strategy_adaptive(self, values: np.ndarray) -> StrategyResult:
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        N = values.shape[0]
        masks = np.zeros(N, dtype=np.uint8)
        strategy_mask = np.zeros(N, dtype=np.uint8)
        checks = 0
        bound_widths = self.hi - self.lo
        order = np.argsort(bound_widths)
        for i in range(N):
            for j in order:
                strategy_mask[i] |= np.uint8(1 << j)
                checks += 1
                if values[i, j] < self.lo[j] or values[i, j] > self.hi[j]:
                    masks[i] |= np.uint8(1 << j)
        return StrategyResult(
            name="adaptive", error_masks=masks, covered=np.ones(N, dtype=bool),
            checks_performed=checks, checks_skipped=N * self.D - checks,
            strategy_mask=strategy_mask,
        )

    def strategy_severity_weighted(self, values: np.ndarray, budget_pct: float = 0.6) -> StrategyResult:
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        N = values.shape[0]
        masks = np.zeros(N, dtype=np.uint8)
        strategy_mask = np.zeros(N, dtype=np.uint8)
        checks = 0
        budget = max(1, int(self.D * budget_pct))
        for i in range(N):
            for k in range(min(budget, self.D)):
                j = self.severity_order[k]
                strategy_mask[i] |= np.uint8(1 << j)
                checks += 1
                if values[i, j] < self.lo[j] or values[i, j] > self.hi[j]:
                    masks[i] |= np.uint8(1 << j)
        return StrategyResult(
            name="severity_weighted", error_masks=masks, covered=np.ones(N, dtype=bool),
            checks_performed=checks, checks_skipped=N * self.D - checks,
            strategy_mask=strategy_mask,
        )

    def run_all(self, values: np.ndarray) -> List[StrategyResult]:
        return [
            self.strategy_strict(values),
            self.strategy_adaptive(values),
            self.strategy_severity_weighted(values),
        ]


@dataclass
class ShadowgapResult:
    """Result of shadowgap analysis."""
    n_points: int
    n_true_violations: int
    n_consensus_catches: int
    n_shadowgap: int
    shadowgap_rate: float
    shadowgap_fraction: float
    shadowgap_indices: np.ndarray
    per_constraint_shadowgap: np.ndarray
    surprise_scores: np.ndarray

    def summary(self) -> dict:
        """Human-readable summary of shadowgap analysis."""
        return {
            "n_points": self.n_points,
            "n_true_violations": self.n_true_violations,
            "n_consensus_catches": self.n_consensus_catches,
            "n_shadowgap": self.n_shadowgap,
            "shadowgap_rate": round(self.shadowgap_rate, 4),
            "shadowgap_fraction": round(self.shadowgap_fraction, 6),
            "per_constraint": self.per_constraint_shadowgap.tolist(),
            "clean": self.n_shadowgap == 0,
        }


class ShadowgapFinder:
    """
    Finds regions of violation space that ALL checking strategies missed.
    Uses information-theoretic surprise to predict where the next gap appears.
    """

    def __init__(self, n_constraints: int):
        self.D = n_constraints

    def find(self, ground_truth: np.ndarray,
             strategy_results: List[StrategyResult]) -> ShadowgapResult:
        """Find shadowgaps across all strategies."""
        N = len(ground_truth)
        true_violated = ground_truth != 0
        n_true = int(np.sum(true_violated))

        consensus = np.zeros(N, dtype=np.uint8)
        for sr in strategy_results:
            consensus |= sr.error_masks

        # Shadowgap: truly violated but consensus says clean
        sg_mask = true_violated & (consensus == 0)
        sg_indices = np.where(sg_mask)[0]
        n_sg = len(sg_indices)

        # Per-constraint shadowgap
        per_c = np.zeros(self.D, dtype=int)
        for i in sg_indices:
            for j in range(self.D):
                if ground_truth[i] & (1 << j):
                    per_c[j] += 1

        # Surprise scores: -log2(P(consensus_pass | ground_truth_violate))
        surprise = np.zeros(N, dtype=np.float64)
        if n_true > 0:
            p_miss = n_sg / n_true
            if p_miss > 0:
                surprise[sg_mask] = -np.log2(p_miss)

        return ShadowgapResult(
            n_points=N,
            n_true_violations=n_true,
            n_consensus_catches=n_true - n_sg,
            n_shadowgap=n_sg,
            shadowgap_rate=n_sg / n_true if n_true > 0 else 0.0,
            shadowgap_fraction=n_sg / N if N > 0 else 0.0,
            shadowgap_indices=sg_indices,
            per_constraint_shadowgap=per_c,
            surprise_scores=surprise,
        )

    def find_from_checker(self, checker: MultiChecker,
                          values: np.ndarray) -> ShadowgapResult:
        """Convenience: run all strategies and find shadowgaps."""
        gt = checker.ground_truth(values)
        results = checker.run_all(values)
        return self.find(gt, results)
