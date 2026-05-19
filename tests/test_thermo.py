"""Tests for flux_lib.thermo — thermodynamic analysis."""

import math
import numpy as np
import pytest
from flux_lib.thermo import (
    violation_entropy, normalized_entropy, partition_function,
    detect_phase_transition, ThermoEngine,
)


class TestViolationEntropy:
    def test_zero_violated(self):
        assert violation_entropy(8, 0) == 0.0

    def test_all_violated(self):
        assert violation_entropy(8, 8) == 0.0

    def test_half_violated(self):
        e = violation_entropy(8, 4)
        assert e > 0  # maximum entropy state

    def test_normalization(self):
        e4 = violation_entropy(8, 4, base=2)
        e3 = violation_entropy(8, 3, base=2)
        assert e4 > e3  # N/2 has higher entropy

    def test_nats(self):
        e = violation_entropy(8, 4, base=math.e)
        assert e > 0

    def test_invalid_n_violated(self):
        with pytest.raises(ValueError):
            violation_entropy(5, -1)
        with pytest.raises(ValueError):
            violation_entropy(5, 6)


class TestNormalizedEntropy:
    def test_zero(self):
        assert normalized_entropy(8, 0) == 0.0

    def test_in_range(self):
        ne = normalized_entropy(8, 4)
        assert 0 <= ne <= 1

    def test_empty(self):
        assert normalized_entropy(0, 0) == 0.0


class TestPartitionFunction:
    def test_basic(self):
        r = partition_function(np.array([1.0, 2.0]))
        assert r.Z > 0
        assert r.free_energy < 0  # F = -kT ln(Z), Z > 1
        assert r.mean_energy >= 0
        assert r.entropy >= 0
        assert r.specific_heat >= 0
        assert len(r.violation_probabilities) == 2

    def test_empty_weights(self):
        r = partition_function(np.array([]))
        assert r.Z == 1.0

    def test_temperature_zero_raises(self):
        with pytest.raises(ValueError):
            partition_function(np.array([1.0]), temperature=0)

    def test_high_temperature(self):
        # At high T, all probabilities approach 0.5
        r = partition_function(np.array([1.0, 1.0]), temperature=1000)
        np.testing.assert_allclose(r.violation_probabilities, 0.5, atol=0.01)


class TestPhaseTransition:
    def test_sharp_transition(self):
        rates = [0.01, 0.02, 0.03, 0.05, 0.8, 0.95]
        result = detect_phase_transition(rates, threshold=1.0)
        assert result.is_transition_detected
        assert result.critical_index >= 3

    def test_no_transition(self):
        rates = [0.01, 0.02, 0.03, 0.04]
        result = detect_phase_transition(rates, threshold=10.0)
        assert not result.is_transition_detected

    def test_too_few_points(self):
        result = detect_phase_transition([0.5])
        assert not result.is_transition_detected


class TestThermoEngine:
    def test_basic(self):
        engine = ThermoEngine([1.0, 2.0, 0.5])
        p = engine.partition()
        assert p.Z > 0

    def test_entropy(self):
        engine = ThermoEngine([1.0, 2.0])
        e = engine.entropy(1)
        assert e > 0

    def test_ideal_gas_check(self):
        engine = ThermoEngine([1.0, 2.0])
        assert engine.ideal_gas_check()

    def test_temperature(self):
        engine = ThermoEngine([1.0, 2.0, 3.0])
        t = engine.temperature(np.array([0.5, 1.0]), 1)
        assert t > 0

    def test_summary(self):
        engine = ThermoEngine([1.0, 2.0])
        s = engine.summary()
        assert "Z" in s
        assert "ideal_gas" in s
        assert s["ideal_gas"] is True
