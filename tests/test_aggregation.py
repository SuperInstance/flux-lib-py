"""Tests for aggregation utilities — check_and_aggregate."""

import pytest
from flux_lib import ConstraintEngine


class TestCheckAndAggregate:
    def _make_engine(self):
        return ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -40, "hi": 150, "name": "b"},
            {"lo": 0, "hi": 8000, "name": "c"},
        ])

    def test_empty_batch(self):
        eng = self._make_engine()
        result = eng.check_and_aggregate([])
        assert result["total_readings"] == 0
        assert result["total_violations"] == 0
        assert result["violation_rate"] == 0.0
        assert result["worst_reading"] is None

    def test_all_pass(self):
        eng = self._make_engine()
        batch = [[50, 90, 3000], [10, 0, 1000], [99, 149, 7999]]
        result = eng.check_and_aggregate(batch)
        assert result["total_readings"] == 3
        assert result["total_violations"] == 0
        assert result["violation_rate"] == 0.0
        assert result["worst_reading"][1].passed

    def test_mixed_violations(self):
        eng = self._make_engine()
        batch = [
            [50, 90, 3000],     # all pass
            [200, 90, 3000],    # a violates
            [50, 999, 3000],    # b violates
            [50, 90, 99999],    # c violates
        ]
        result = eng.check_and_aggregate(batch)
        assert result["total_readings"] == 4
        assert result["total_violations"] == 3
        assert result["violation_rate"] == 3 / (4 * 3)  # 3 violations / 12 checks

    def test_per_constraint_rate(self):
        eng = self._make_engine()
        batch = [
            [200, 90, 3000],    # a violates
            [200, 90, 3000],    # a violates
            [50, 90, 3000],     # all pass
        ]
        result = eng.check_and_aggregate(batch)
        assert result["per_constraint_violation_rate"]["a"] == 2 / 3
        assert result["per_constraint_violation_rate"]["b"] == 0.0
        assert result["per_constraint_violation_rate"]["c"] == 0.0

    def test_worst_reading(self):
        eng = self._make_engine()
        batch = [
            [50, 90, 3000],       # 0 violations
            [200, 999, 99999],    # 3 violations — worst
            [200, 90, 3000],      # 1 violation
        ]
        result = eng.check_and_aggregate(batch)
        idx, check_result = result["worst_reading"]
        assert idx == 1
        assert check_result.violated_count == 3

    def test_severity_breakdown(self):
        eng = self._make_engine()
        batch = [
            [50, 90, 3000],     # PASS
            [200, 90, 3000],    # WARNING (default severity)
        ]
        result = eng.check_and_aggregate(batch)
        assert result["severity_breakdown"]["PASS"] == 1
        assert result["severity_breakdown"]["WARNING"] >= 1

    def test_all_violate(self):
        eng = self._make_engine()
        batch = [
            [200, 999, 99999],  # all 3 violate
            [-1, -999, -1],     # all 3 violate
        ]
        result = eng.check_and_aggregate(batch)
        assert result["total_violations"] == 6
        assert result["violation_rate"] == 1.0

    def test_nan_violations_counted(self):
        eng = self._make_engine()
        batch = [
            [float("nan"), 90, 3000],  # a violates via NaN
        ]
        result = eng.check_and_aggregate(batch)
        assert result["total_violations"] == 1
        assert result["per_constraint_violation_rate"]["a"] == 1.0

    def test_with_preset_engine(self):
        eng = ConstraintEngine.from_preset("automotive_can")
        # All 8 sensors in range
        batch = [
            [3000, 65, 90, 40, 10, 45, 12.5, 50],
            [3000, 65, 90, 40, 10, 45, 12.5, 50],
        ]
        result = eng.check_and_aggregate(batch)
        assert result["total_readings"] == 2
        assert result["total_violations"] == 0
