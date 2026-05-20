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


class TestCheckVector:
    def test_vector_all_pass(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 8000, "name": "rpm"},
            {"lo": -40, "hi": 150, "name": "coolant"},
        ])
        r = eng.check_vector([3000, 90])
        assert r.passed
        assert r.error_mask == 0
        assert r.violated_count == 0

    def test_vector_one_fails(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 8000, "name": "rpm"},
            {"lo": -40, "hi": 150, "name": "coolant"},
        ])
        r = eng.check_vector([3000, 999])  # coolant violates
        assert not r.passed
        assert r.error_mask == 0b10  # bit 1 set
        assert r.violated_count == 1
        assert r.violations[1].name == "coolant"

    def test_vector_both_fail(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 8000, "name": "rpm"},
            {"lo": -40, "hi": 150, "name": "coolant"},
        ])
        r = eng.check_vector([99999, -999])
        assert not r.passed
        assert r.error_mask == 0b11
        assert r.violated_count == 2

    def test_vector_nan_violates(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": 0, "hi": 100, "name": "b"},
        ])
        r = eng.check_vector([50, float("nan")])
        assert not r.passed
        assert r.error_mask == 0b10  # only b violates

    def test_vector_wrong_length_raises(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": 0, "hi": 100, "name": "b"},
        ])
        with pytest.raises(ValueError, match="Expected 2"):
            eng.check_vector([50])
        with pytest.raises(ValueError, match="Expected 2"):
            eng.check_vector([50, 50, 50])

    def test_vector_boundary_inclusive(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -40, "hi": 150, "name": "b"},
        ])
        r = eng.check_vector([0, -40])
        assert r.passed
        r = eng.check_vector([100, 150])
        assert r.passed

    def test_vector_severity(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a", "severity": 1},
            {"lo": 0, "hi": 100, "name": "b", "severity": 3},
        ])
        r = eng.check_vector([50, 999])
        assert r.severity == Severity.CRITICAL  # from constraint b

    def test_vector_with_preset(self):
        eng = ConstraintEngine.from_preset("automotive_can")
        # 8 values for 8 constraints — all in range
        vals = [3000, 65, 90, 40, 10, 45, 12.5, 50]
        r = eng.check_vector(vals)
        assert r.passed
        # Last value out of range
        vals2 = [3000, 65, 90, 40, 10, 45, 12.5, 999]
        r2 = eng.check_vector(vals2)
        assert not r2.passed


class TestCheckVectorBatch:
    def test_batch_vector_basic(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -40, "hi": 150, "name": "b"},
        ])
        samples = np.array([
            [50, 90],      # both pass
            [101, 90],     # a violates
            [50, 999],     # b violates
            [101, 999],    # both violate
        ])
        masks = eng.check_vector_batch(samples)
        np.testing.assert_array_equal(masks, [0, 1, 2, 3])

    def test_batch_vector_wrong_cols_raises(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -40, "hi": 150, "name": "b"},
        ])
        with pytest.raises(ValueError, match="Expected 2 columns"):
            eng.check_vector_batch(np.array([[50, 90, 10]]))

    def test_batch_vector_1d(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -40, "hi": 150, "name": "b"},
        ])
        masks = eng.check_vector_batch(np.array([50, 90]))
        assert masks.shape == (1,)
        assert masks[0] == 0

    def test_batch_vector_nan(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -40, "hi": 150, "name": "b"},
        ])
        samples = np.array([
            [50, float("nan")],
            [float("nan"), 90],
        ])
        masks = eng.check_vector_batch(samples)
        assert masks[0] == 2  # b violates
        assert masks[1] == 1  # a violates
