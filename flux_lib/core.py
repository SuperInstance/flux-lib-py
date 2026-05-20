"""
flux_lib.core — ConstraintEngine: exact bound checking, zero false negatives.

INVARIANT: A value outside bounds is ALWAYS detected. NaN always violates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from flux_lib.presets import PRESETS, get_preset

Number = Union[int, float]


# ── Severity ────────────────────────────────────────────────

class Severity(IntEnum):
    PASS = 0
    CAUTION = 1
    WARNING = 2
    CRITICAL = 3


_SEVERITY_TABLE = [
    Severity.PASS,     # 0 violations
    Severity.CAUTION,  # 1
    Severity.CAUTION,  # 2
    Severity.WARNING,  # 3
    Severity.WARNING,  # 4
    Severity.CRITICAL, # 5
    Severity.CRITICAL, # 6
    Severity.CRITICAL, # 7
    Severity.CRITICAL, # 8
]


def _severity_for_count(n: int) -> Severity:
    return _SEVERITY_TABLE[n] if n < len(_SEVERITY_TABLE) else Severity.CRITICAL


# ── Data classes ────────────────────────────────────────────

@dataclass(frozen=True)
class ConstraintDef:
    lo: float
    hi: float
    name: str
    severity: Severity = Severity.WARNING

    def __post_init__(self):
        if self.lo > self.hi:
            raise ValueError(f"Constraint '{self.name}': lo ({self.lo}) > hi ({self.hi})")


@dataclass
class ViolationDetail:
    name: str
    lo: float
    hi: float
    value: float
    passed: bool
    lo_violated: bool
    hi_violated: bool


@dataclass
class CheckResult:
    """Result of checking values against constraints."""
    error_mask: int = 0
    severity: Severity = Severity.PASS
    violated_lo: int = 0
    violated_hi: int = 0
    violated_count: int = 0
    violations: List[ViolationDetail] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.error_mask == 0

    def to_dict(self) -> dict:
        return {
            "error_mask": self.error_mask,
            "severity": int(self.severity),
            "severity_name": self.severity.name,
            "violated_count": self.violated_count,
            "passed": self.passed,
            "violations": [
                {"name": v.name, "lo": v.lo, "hi": v.hi, "value": v.value, "passed": v.passed}
                for v in self.violations
            ],
        }


# ── ConstraintEngine ────────────────────────────────────────

class ConstraintEngine:
    """
    Exact constraint engine — zero false negatives.

    The error mask is uint8, supporting up to **8 constraints**.
    For more constraints, use multiple engines.

    check()             → CheckResult (one value against ALL constraints)
    check_vector()      → CheckResult (N values against N respective constraints)
    check_vector_batch()→ np.ndarray of uint8 masks (batch vector mode)
    check_mask()        → int (zero-alloc hot path)
    check_batch()       → np.ndarray of uint8 masks
    """

    __slots__ = ("_lo", "_hi", "_names", "_severities", "n", "constraints")

    def __init__(self, constraints: List[Dict]):
        if not constraints:
            raise ValueError("ConstraintEngine requires a non-empty constraints list")
        if len(constraints) > 8:
            raise ValueError("Maximum 8 constraints (error_mask is uint8)")

        self._lo = tuple(float(c["lo"]) for c in constraints)
        self._hi = tuple(float(c["hi"]) for c in constraints)
        self._names = tuple(c.get("name", f"C{i}") for i, c in enumerate(constraints))
        self._severities = tuple(
            Severity(c.get("severity", 2)) for c in constraints
        )
        self.n = len(constraints)

        for i in range(self.n):
            if self._lo[i] > self._hi[i]:
                raise ValueError(
                    f"Constraint '{self._names[i]}': lo ({self._lo[i]}) > hi ({self._hi[i]})"
                )

        self.constraints = [
            ConstraintDef(
                lo=self._lo[i], hi=self._hi[i],
                name=self._names[i], severity=self._severities[i],
            )
            for i in range(self.n)
        ]

    # ── Zero-alloc hot path ─────────────────────────────────

    def check_mask(self, value: Number) -> int:
        """Check value. Returns error_mask (0 = all pass). Zero allocations."""
        v = float(value)
        if v != v:  # NaN
            return (1 << self.n) - 1
        mask = 0
        for i in range(self.n):
            if v < self._lo[i] or v > self._hi[i]:
                mask |= (1 << i)
        return mask

    # ── Full check (returns CheckResult) ─────────────────────

    def check(self, value: Number) -> CheckResult:
        """Check value. Returns CheckResult with .passed, .severity, .violations."""
        v = float(value)
        is_nan = v != v
        mask = 0
        lo_mask = 0
        hi_mask = 0
        violations: list[ViolationDetail] = []

        for i in range(self.n):
            if is_nan:
                lo_f = hi_f = True
            else:
                lo_f = v < self._lo[i]
                hi_f = v > self._hi[i]
            p = not lo_f and not hi_f
            if not p:
                mask |= (1 << i)
            if lo_f:
                lo_mask |= (1 << i)
            if hi_f:
                hi_mask |= (1 << i)
            violations.append(ViolationDetail(
                name=self._names[i], lo=self._lo[i], hi=self._hi[i], value=v,
                passed=p, lo_violated=lo_f, hi_violated=hi_f,
            ))

        vc = bin(mask).count("1")
        # Severity = max of violated constraint severities, or count-based
        worst = Severity.PASS
        for i in range(self.n):
            if mask & (1 << i):
                worst = max(worst, self._severities[i])
        if worst == Severity.PASS and vc > 0:
            worst = _severity_for_count(vc)

        return CheckResult(
            error_mask=mask,
            severity=worst if vc > 0 else Severity.PASS,
            violated_lo=lo_mask,
            violated_hi=hi_mask,
            violated_count=vc,
            violations=violations,
        )

    # ── Vector: N values → N respective constraints ──────────

    def check_vector(self, values: list[float]) -> CheckResult:
        """Check N values against N respective constraints.

        value[i] is checked against constraint[i].
        Returns a CheckResult where bit i is set if value[i]
        violates its corresponding constraint.
        """
        if len(values) != self.n:
            raise ValueError(
                f"Expected {self.n} values (one per constraint), got {len(values)}"
            )
        mask = 0
        lo_mask = 0
        hi_mask = 0
        violations: list[ViolationDetail] = []

        for i in range(self.n):
            v = float(values[i])
            is_nan = v != v
            if is_nan:
                lo_f = hi_f = True
            else:
                lo_f = v < self._lo[i]
                hi_f = v > self._hi[i]
            p = not lo_f and not hi_f
            if not p:
                mask |= (1 << i)
            if lo_f:
                lo_mask |= (1 << i)
            if hi_f:
                hi_mask |= (1 << i)
            violations.append(ViolationDetail(
                name=self._names[i], lo=self._lo[i], hi=self._hi[i], value=v,
                passed=p, lo_violated=lo_f, hi_violated=hi_f,
            ))

        vc = bin(mask).count("1")
        worst = Severity.PASS
        for i in range(self.n):
            if mask & (1 << i):
                worst = max(worst, self._severities[i])
        if worst == Severity.PASS and vc > 0:
            worst = _severity_for_count(vc)

        return CheckResult(
            error_mask=mask,
            severity=worst if vc > 0 else Severity.PASS,
            violated_lo=lo_mask,
            violated_hi=hi_mask,
            violated_count=vc,
            violations=violations,
        )

    def check_vector_batch(self, values_array: np.ndarray) -> np.ndarray:
        """Vectorized batch check_vector.

        values_array shape: (N_samples, N_constraints) or (N_constraints,).
        value[j, i] is checked against constraint[i].
        Returns np.ndarray of uint8 error_masks with shape (N_samples,).
        """
        vals = np.asarray(values_array, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        if vals.shape[1] != self.n:
            raise ValueError(
                f"Expected {self.n} columns (one per constraint), got {vals.shape[1]}"
            )
        N = vals.shape[0]
        masks = np.zeros(N, dtype=np.uint8)
        nan_rows = np.any(np.isnan(vals), axis=1)
        for i in range(self.n):
            col = vals[:, i]
            col_nan = np.isnan(col)
            violated = col_nan | (col < self._lo[i]) | (col > self._hi[i])
            masks[violated] |= np.uint8(1 << i)
        return masks

    # ── Batch: numpy vectorized ──────────────────────────────

    def check_batch(self, values) -> np.ndarray:
        """Vectorized batch check. Returns np.ndarray of uint8 error_masks."""
        vals = np.asarray(values, dtype=np.float64)
        flat = vals.ravel()
        masks = np.zeros(len(flat), dtype=np.uint8)
        nan_mask = np.isnan(flat)
        masks[nan_mask] = np.uint8((1 << self.n) - 1)
        valid = ~nan_mask
        for i in range(self.n):
            violated = valid & ((flat < self._lo[i]) | (flat > self._hi[i]))
            masks[violated] |= np.uint8(1 << i)
        return masks.reshape(vals.shape)

    # ── Presets ──────────────────────────────────────────────

    @classmethod
    def from_preset(cls, name: str) -> "ConstraintEngine":
        """Create engine from industry preset."""
        return cls(get_preset(name))

    @classmethod
    def available_presets(cls) -> List[str]:
        return sorted(PRESETS.keys())

    # ── Benchmark ────────────────────────────────────────────

    def benchmark(self, iterations: int = 1_000_000) -> float:
        """Returns checks/sec for check_mask hot path."""
        t0 = time.perf_counter()
        for i in range(iterations):
            self.check_mask((i % 1000) - 500)
        return iterations / (time.perf_counter() - t0)

    def __repr__(self) -> str:
        return f"ConstraintEngine(n={self.n}, constraints={self._names})"
