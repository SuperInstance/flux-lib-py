"""Tests for flux_lib.core — ConstraintEngine."""

import numpy as np
import pytest
from flux_lib.core import ConstraintEngine, CheckResult, Severity


class TestConstraintEngineBasic:
    def test_single_constraint_pass(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "temp"}])
        r = eng.check(50)
        assert r.passed
        assert r.error_mask == 0

    def test_single_constraint_fail_high(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "temp"}])
        r = eng.check(101)
        assert not r.passed
        assert r.error_mask == 1

    def test_single_constraint_fail_low(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "temp"}])
        r = eng.check(-1)
        assert not r.passed
        assert r.violated_lo == 1

    def test_boundary_inclusive(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        assert eng.check(0).passed
        assert eng.check(100).passed

    def test_nan_violates_all(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        r = eng.check(float("nan"))
        assert not r.passed
        assert r.error_mask == 1

    def test_inf_violates(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        assert not eng.check(float("inf")).passed
        assert not eng.check(float("-inf")).passed


class TestConstraintEngineMulti:
    def test_two_constraints_both_pass(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -10, "hi": 10, "name": "b"},
        ])
        r = eng.check(5)
        assert r.passed
        assert r.error_mask == 0

    def test_two_constraints_one_fails(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 50, "name": "a"},
            {"lo": -10, "hi": 10, "name": "b"},
        ])
        r = eng.check(60)
        assert not r.passed
        assert r.error_mask == 0b11  # both violated (60 > 50 and 60 > 10)

    def test_two_constraints_both_fail(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 50, "name": "a"},
            {"lo": 60, "hi": 100, "name": "b"},
        ])
        r = eng.check(55)
        assert not r.passed
        assert r.error_mask == 0b11


class TestConstraintEngineMask:
    def test_check_mask_zero_alloc(self):
        eng = ConstraintEngine([{"lo": -40, "hi": 150, "name": "t"}])
        assert eng.check_mask(50) == 0
        assert eng.check_mask(151) == 1
        assert eng.check_mask(-41) == 1

    def test_check_mask_nan(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        assert eng.check_mask(float("nan")) == 1


class TestConstraintEngineBatch:
    def test_batch_basic(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        arr = np.array([50, 101, -1, 0, 100])
        masks = eng.check_batch(arr)
        np.testing.assert_array_equal(masks, [0, 1, 1, 0, 0])

    def test_batch_nan(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        arr = np.array([50, float("nan")])
        masks = eng.check_batch(arr)
        np.testing.assert_array_equal(masks, [0, 1])

    def test_batch_shape_preserved(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        arr = np.array([[50, 101], [-1, 50]])
        masks = eng.check_batch(arr)
        assert masks.shape == (2, 2)


class TestConstraintEngineValidation:
    def test_empty_constraints_raises(self):
        with pytest.raises(ValueError):
            ConstraintEngine([])

    def test_max_8_constraints(self):
        with pytest.raises(ValueError):
            ConstraintEngine([{"lo": 0, "hi": 1, "name": f"c{i}"} for i in range(9)])

    def test_lo_gt_hi_raises(self):
        with pytest.raises(ValueError):
            ConstraintEngine([{"lo": 100, "hi": 0, "name": "bad"}])


class TestConstraintEnginePresets:
    def test_preset_automotive(self):
        eng = ConstraintEngine.from_preset("automotive_can")
        assert eng.n == 8

    def test_preset_medical(self):
        eng = ConstraintEngine.from_preset("medical_fhir")
        r = eng.check(37.0)  # body temp in range
        # body_temp_c passes, but other constraints checked against same value
        assert isinstance(r, CheckResult)

    def test_preset_not_found(self):
        with pytest.raises(KeyError):
            ConstraintEngine.from_preset("nonexistent")

    def test_all_presets_load(self):
        for name in ConstraintEngine.available_presets():
            eng = ConstraintEngine.from_preset(name)
            assert eng.n > 0

    def test_available_presets_count(self):
        assert len(ConstraintEngine.available_presets()) == 10


class TestConstraintEngineSeverity:
    def test_severity_from_constraint(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "x", "severity": 3},
        ])
        r = eng.check(101)
        assert r.severity == Severity.CRITICAL

    def test_severity_pass(self):
        eng = ConstraintEngine([{"lo": 0, "hi": 100, "name": "x"}])
        assert eng.check(50).severity == Severity.PASS
