"""
flux_lib.drift — Time-series drift detection for sensor readings.

Sliding window analysis that detects gradual drift toward constraint
boundaries and forecasts when violations are likely to occur.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


class DriftDetector:
    """Detect gradual drift in sensor readings over time.

    Maintains a sliding window of recent readings and uses linear regression
    on each sensor dimension to detect drift toward constraint boundaries.

    Usage:
        det = DriftDetector(window_size=100)
        for reading in sensor_stream:
            det.add(reading)
            if det.n >= 50:
                drift = det.detect_drift(bounds=engine)
    """

    def __init__(self, window_size: int = 100):
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        self.window: list[list[float]] = []
        self.window_size = window_size

    @property
    def n(self) -> int:
        """Number of readings currently in the window."""
        return len(self.window)

    def add(self, values: list[float]) -> None:
        """Add a reading to the sliding window."""
        self.window.append(list(values))
        if len(self.window) > self.window_size:
            self.window.pop(0)

    def _trends(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-sensor linear trend (slope, intercept) over the window."""
        data = np.array(self.window, dtype=float)  # shape (T, D)
        T, D = data.shape
        x = np.arange(T, dtype=float)
        slopes = np.zeros(D)
        intercepts = np.zeros(D)
        for d in range(D):
            if T < 2:
                continue
            # Simple linear regression: slope = cov(x,y) / var(x)
            xm = x.mean()
            ym = data[:, d].mean()
            dx = x - xm
            dy = data[:, d] - ym
            var_x = (dx * dx).sum()
            if var_x > 0:
                slopes[d] = (dx * dy).sum() / var_x
            intercepts[d] = ym - slopes[d] * xm
        return slopes, intercepts

    if TYPE_CHECKING:
        from flux_lib.core import ConstraintEngine

    @staticmethod
    def _parse_bounds(
        bounds: Union[
            "ConstraintEngine",
            List[Tuple[str, float, float]],
            List[Tuple[float, float]],
            None,
        ],
        n_dims: int,
    ) -> tuple[list[str], list[tuple[float, float]] | None]:
        """Normalise *bounds* into (names, lo_hi_pairs).

        Accepts:
          - ConstraintEngine   → names + lo/hi from constraints
          - [(name, lo, hi),…] → explicit names
          - [(lo, hi),…]       → sensor_N names (backwards compat)
          - None               → sensor_N names, no bounds
        """
        if bounds is None:
            return [f"sensor_{d}" for d in range(n_dims)], None

        # ConstraintEngine
        if hasattr(bounds, "get_bounds") and hasattr(bounds, "n"):
            engine: ConstraintEngine = bounds  # type: ignore[assignment]
            names = [engine.constraints[i].name for i in range(min(engine.n, n_dims))]
            # Pad if engine has fewer constraints than dims
            while len(names) < n_dims:
                names.append(f"sensor_{len(names)}")
            lo_hi = [(engine.constraints[i].lo, engine.constraints[i].hi) for i in range(min(engine.n, n_dims))]
            return names, lo_hi

        # Peek at first element to distinguish (name, lo, hi) vs (lo, hi)
        first = bounds[0]  # type: ignore[index]
        if len(first) == 3:
            # (name, lo, hi)
            names = [str(b[0]) for b in bounds]  # type: ignore[index]
            lo_hi = [(float(b[1]), float(b[2])) for b in bounds]  # type: ignore[index]
            return names, lo_hi
        else:
            # (lo, hi) — backwards compat
            names = [f"sensor_{d}" for d in range(len(bounds))]  # type: ignore[arg-type]
            lo_hi = [(float(b[0]), float(b[1])) for b in bounds]  # type: ignore[index]
            return names, lo_hi

    def detect_drift(
        self,
        bounds: Union[
            "ConstraintEngine",
            List[Tuple[str, float, float]],
            List[Tuple[float, float]],
            None,
        ] = None,
    ) -> dict:
        """Check if values are drifting toward bounds.

        Args:
            bounds: One of:
                - ConstraintEngine: uses constraint names and bounds
                - list of (name, lo, hi) tuples: explicit names per sensor
                - list of (lo, hi) tuples: backwards compat (sensor_N names)
                - None: no bounds, sensor_N names

        Returns:
            dict with keys:
                drifting: bool — any sensor drifting significantly
                per_sensor: {name: {direction, rate}}
                time_to_violation: {name: float|None} (only if bounds given)
        """
        if len(self.window) < 3:
            return {"drifting": False, "per_sensor": {}, "time_to_violation": {}}

        data = np.array(self.window, dtype=float)
        T, D = data.shape
        slopes, _ = self._trends()

        names, lo_hi_pairs = self._parse_bounds(bounds, D)

        # Standard deviation per sensor for significance threshold
        stds = data.std(axis=0)
        threshold = 0.1 * stds / T if T > 0 else np.zeros(D)

        per_sensor: Dict[str, dict] = {}
        time_to_violation: Dict[str, dict] = {}
        any_drifting = False

        last_reading = data[-1]

        for d in range(D):
            name = names[d] if d < len(names) else f"sensor_{d}"
            slope = slopes[d]
            abs_slope = abs(slope)
            direction = "stable"
            rate = 0.0

            if abs_slope > threshold[d] and threshold[d] > 0:
                if slope > 0:
                    direction = "toward_hi"
                else:
                    direction = "toward_lo"
                rate = float(slope)
                any_drifting = True

            per_sensor[name] = {"direction": direction, "rate": rate}

            # Time-to-violation estimate
            if lo_hi_pairs is not None and d < len(lo_hi_pairs):
                lo, hi = lo_hi_pairs[d]
                ttv = None
                if direction == "toward_hi" and slope > 0:
                    readings_to_violation = (hi - last_reading[d]) / slope
                    ttv = max(0.0, float(readings_to_violation)) if readings_to_violation > 0 else None
                elif direction == "toward_lo" and slope < 0:
                    readings_to_violation = (lo - last_reading[d]) / slope
                    ttv = max(0.0, float(readings_to_violation)) if readings_to_violation > 0 else None
                time_to_violation[name] = ttv

        return {
            "drifting": any_drifting,
            "per_sensor": per_sensor,
            "time_to_violation": time_to_violation,
        }

    def forecast(self, n_ahead: int = 10) -> list[list[float]]:
        """Simple linear forecast of next N readings based on trend.

        Uses linear regression on the sliding window to project forward.
        Returns list of N forecasted readings (each a list of sensor values).
        """
        if len(self.window) < 2 or n_ahead < 1:
            return []

        slopes, intercepts = self._trends()
        data = np.array(self.window, dtype=float)
        T = data.shape[0]

        forecasts = []
        for step in range(1, n_ahead + 1):
            t_future = T - 1 + step
            forecast = [float(intercepts[d] + slopes[d] * t_future)
                        for d in range(data.shape[1])]
            forecasts.append(forecast)
        return forecasts

    def reset(self) -> None:
        """Clear the sliding window."""
        self.window.clear()
