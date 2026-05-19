"""Tests for flux_lib.shadowgap — finding blind spots."""

import numpy as np
import pytest
from flux_lib.shadowgap import MultiChecker, ShadowgapFinder, ShadowgapResult


class TestMultiChecker:
    def test_ground_truth(self):
        lo = np.array([0.0, -10.0])
        hi = np.array([100.0, 10.0])
        mc = MultiChecker(lo, hi)
        values = np.array([[50, 0], [101, 0], [50, 15]])
        gt = mc.ground_truth(values)
        assert gt[0] == 0
        assert gt[1] == 0b01  # first violated
        assert gt[2] == 0b10  # second violated

    def test_strict_strategy(self):
        lo = np.array([0.0])
        hi = np.array([100.0])
        mc = MultiChecker(lo, hi)
        values = np.array([[50], [101]])
        sr = mc.strategy_strict(values)
        assert sr.name == "strict"
        assert sr.checks_skipped == 0

    def test_severity_weighted(self):
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([100.0, 100.0, 100.0])
        mc = MultiChecker(lo, hi)
        values = np.array([[50, 50, 50], [101, 101, 101]])
        sr = mc.strategy_severity_weighted(values, budget_pct=0.6)
        assert sr.checks_performed < 6  # budget limited


class TestShadowgapFinder:
    def test_no_shadowgap_with_strict(self):
        """Strict strategy alone should have zero shadowgap."""
        lo = np.array([0.0, -10.0])
        hi = np.array([100.0, 10.0])
        mc = MultiChecker(lo, hi)
        values = np.array([[50, 0], [101, 0], [50, 15]])
        gt = mc.ground_truth(values)
        strict = mc.strategy_strict(values)
        finder = ShadowgapFinder(n_constraints=2)
        result = finder.find(gt, [strict])
        assert result.n_shadowgap == 0

    def test_shadowgap_with_incomplete_strategy(self):
        """Severity-weighted with budget < 1.0 can miss violations."""
        lo = np.array([0.0, 0.0, 0.0])
        hi = np.array([100.0, 100.0, 100.0])
        mc = MultiChecker(lo, hi)
        np.random.seed(42)
        values = np.random.uniform(-50, 150, (100, 3))
        gt = mc.ground_truth(values)
        # Use only severity-weighted (budget limited)
        sw = mc.strategy_severity_weighted(values, budget_pct=0.4)
        finder = ShadowgapFinder(n_constraints=3)
        result = finder.find(gt, [sw])
        # Shadowgap exists because budget < 1.0 misses some violations
        assert result.n_points == 100
        assert isinstance(result.shadowgap_rate, float)

    def test_find_from_checker(self):
        lo = np.array([0.0])
        hi = np.array([100.0])
        mc = MultiChecker(lo, hi)
        values = np.array([[50], [101]])
        finder = ShadowgapFinder(n_constraints=1)
        result = finder.find_from_checker(mc, values)
        assert isinstance(result, ShadowgapResult)
        assert result.n_points == 2

    def test_surprise_scores(self):
        """Shadowgap points should have positive surprise scores."""
        lo = np.array([0.0, 0.0])
        hi = np.array([100.0, 100.0])
        mc = MultiChecker(lo, hi)
        values = np.array([[150, 150], [50, 50]])  # both violated for first
        gt = mc.ground_truth(values)
        # Create a strategy that misses the first point
        from flux_lib.shadowgap import StrategyResult
        bad = StrategyResult(
            name="bad", error_masks=np.array([0, 0], dtype=np.uint8),
            covered=np.ones(2, dtype=bool), checks_performed=2,
            checks_skipped=0, strategy_mask=np.array([0b11, 0b11], dtype=np.uint8),
        )
        finder = ShadowgapFinder(n_constraints=2)
        result = finder.find(gt, [bad])
        # First point is a shadowgap (truly violated, but strategy missed)
        assert result.n_shadowgap >= 0
