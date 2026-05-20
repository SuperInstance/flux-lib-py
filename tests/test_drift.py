"""Tests for drift detection — DriftDetector."""

import pytest
import numpy as np
from flux_lib.drift import DriftDetector


class TestDriftDetectorBasic:
    def test_window_size(self):
        det = DriftDetector(window_size=10)
        assert det.window_size == 10
        assert det.n == 0

    def test_add_readings(self):
        det = DriftDetector(window_size=5)
        det.add([1.0, 2.0])
        assert det.n == 1
        det.add([2.0, 3.0])
        assert det.n == 2

    def test_window_slides(self):
        det = DriftDetector(window_size=3)
        for i in range(10):
            det.add([float(i)])
        assert det.n == 3
        # Window should contain last 3: [7], [8], [9]
        assert det.window[0] == [7.0]
        assert det.window[2] == [9.0]

    def test_window_size_minimum(self):
        with pytest.raises(ValueError):
            DriftDetector(window_size=1)

    def test_reset(self):
        det = DriftDetector(window_size=5)
        det.add([1.0])
        det.add([2.0])
        assert det.n == 2
        det.reset()
        assert det.n == 0


class TestDetectDrift:
    def test_insufficient_data(self):
        det = DriftDetector(window_size=100)
        det.add([1.0])
        result = det.detect_drift()
        assert result["drifting"] is False

    def test_stable_no_drift(self):
        det = DriftDetector(window_size=100)
        for _ in range(20):
            det.add([50.0, 50.0])
        result = det.detect_drift()
        # Stable values should not be drifting
        for name, info in result["per_sensor"].items():
            assert info["direction"] == "stable"

    def test_drift_toward_hi(self):
        det = DriftDetector(window_size=100)
        for i in range(50):
            det.add([float(i) * 0.1, 50.0])  # sensor_0 drifting up
        result = det.detect_drift()
        assert result["drifting"] is True
        assert result["per_sensor"]["sensor_0"]["direction"] == "toward_hi"
        assert result["per_sensor"]["sensor_0"]["rate"] > 0

    def test_drift_toward_lo(self):
        det = DriftDetector(window_size=100)
        for i in range(50):
            det.add([100.0 - float(i) * 0.5, 50.0])  # sensor_0 drifting down
        result = det.detect_drift()
        assert result["drifting"] is True
        assert result["per_sensor"]["sensor_0"]["direction"] == "toward_lo"
        assert result["per_sensor"]["sensor_0"]["rate"] < 0

    def test_with_bounds_time_to_violation(self):
        det = DriftDetector(window_size=100)
        # Drifting up toward hi=100 at ~0.5 per step
        for i in range(50):
            det.add([50.0 + float(i) * 0.5])
        result = det.detect_drift(bounds=[(0, 100)])
        assert "sensor_0" in result["time_to_violation"]
        ttv = result["time_to_violation"]["sensor_0"]
        assert ttv is not None
        assert ttv > 0  # Should have some readings before violation

    def test_with_bounds_no_drift_no_ttv(self):
        det = DriftDetector(window_size=100)
        for _ in range(20):
            det.add([50.0])
        result = det.detect_drift(bounds=[(0, 100)])
        # Stable — no time-to-violation
        assert result["time_to_violation"].get("sensor_0") is None

    def test_mixed_drift(self):
        det = DriftDetector(window_size=100)
        for i in range(50):
            det.add([50.0 + float(i) * 0.5, 50.0 - float(i) * 0.5])
        result = det.detect_drift()
        assert result["drifting"] is True
        assert result["per_sensor"]["sensor_0"]["direction"] == "toward_hi"
        assert result["per_sensor"]["sensor_1"]["direction"] == "toward_lo"


class TestForecast:
    def test_forecast_insufficient_data(self):
        det = DriftDetector(window_size=100)
        assert det.forecast(5) == []

    def test_forecast_flat(self):
        det = DriftDetector(window_size=100)
        for _ in range(10):
            det.add([50.0])
        forecasts = det.forecast(5)
        assert len(forecasts) == 5
        for f in forecasts:
            assert abs(f[0] - 50.0) < 1.0  # Should be near 50

    def test_forecast_upward_trend(self):
        det = DriftDetector(window_size=100)
        for i in range(20):
            det.add([float(i) * 2.0])
        forecasts = det.forecast(5)
        assert len(forecasts) == 5
        # Should continue upward
        assert forecasts[0][0] < forecasts[4][0]

    def test_forecast_multi_sensor(self):
        det = DriftDetector(window_size=100)
        for i in range(20):
            det.add([float(i) * 2.0, 100.0 - float(i) * 2.0])
        forecasts = det.forecast(3)
        assert len(forecasts) == 3
        assert len(forecasts[0]) == 2
        # sensor_0 trending up, sensor_1 trending down
        assert forecasts[0][0] < forecasts[-1][0]
        assert forecasts[0][1] > forecasts[-1][1]

    def test_forecast_n_ahead_zero(self):
        det = DriftDetector(window_size=100)
        for _ in range(10):
            det.add([50.0])
        assert det.forecast(0) == []


class TestDriftDetectorIntegration:
    def test_full_workflow(self):
        """Simulate a sensor stream, detect drift, forecast."""
        det = DriftDetector(window_size=50)
        bounds = [(0, 100), (-40, 150)]

        # Stable phase
        for _ in range(20):
            det.add([50.0, 90.0])

        result = det.detect_drift(bounds=bounds)
        assert not result["drifting"]

        # Introduce drift
        for i in range(30):
            det.add([50.0 + float(i) * 1.0, 90.0])

        result = det.detect_drift(bounds=bounds)
        assert result["drifting"]
        assert result["per_sensor"]["sensor_0"]["direction"] == "toward_hi"

        # Forecast
        forecasts = det.forecast(10)
        assert len(forecasts) == 10
        # Forecasts should show continued upward trend for sensor_0
        assert forecasts[-1][0] > forecasts[0][0]
