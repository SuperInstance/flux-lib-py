"""
flux_lib.thermo — Thermodynamic analysis of constraint systems.

Partition function Z, temperature, energy, entropy, phase transitions.
Ideal gas law for independent constraints: Z = prod(Z_i).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# ── Entropy ─────────────────────────────────────────────────

def violation_entropy(n_constraints: int, n_violated: int, base: float = 2.0) -> float:
    """Constraint entropy S = log_base(C(N, M)). Bits by default."""
    if n_violated < 0 or n_violated > n_constraints:
        raise ValueError(f"n_violated ({n_violated}) must be in [0, {n_constraints}]")
    if n_violated == 0 or n_violated == n_constraints:
        return 0.0
    log_omega = (
        math.lgamma(n_constraints + 1)
        - math.lgamma(n_violated + 1)
        - math.lgamma(n_constraints - n_violated + 1)
    )
    return log_omega if base == math.e else log_omega / math.log(base)


def normalized_entropy(n_constraints: int, n_violated: int) -> float:
    """Normalized entropy in [0, 1]. Max at M = N/2."""
    if n_constraints == 0:
        return 0.0
    return violation_entropy(n_constraints, n_violated, 2.0) / n_constraints


# ── Partition function ──────────────────────────────────────

@dataclass
class PartitionResult:
    """Thermodynamic quantities from the constraint partition function."""
    Z: float
    free_energy: float
    mean_energy: float
    entropy: float
    specific_heat: float
    violation_probabilities: np.ndarray


def partition_function(weights: np.ndarray,
                       temperature: float = 1.0,
                       k: float = 1.0) -> PartitionResult:
    """
    Compute Z = prod(1 + exp(-w_i / kT)) and derived quantities.

    For binary constraints with weights w_i:
        Z  = prod(1 + exp(-w_i/kT))
        F  = -kT ln(Z)
        E  = sum(w_i * P(v_i))
        S  = (E - F) / T
        C  = sum(w_i^2 * p_i(1-p_i)) / (kT)^2
    """
    w = np.asarray(weights, dtype=float)
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    if len(w) == 0:
        return PartitionResult(1.0, 0.0, 0.0, 0.0, 0.0, np.array([]))

    kT = k * temperature
    boltz = np.exp(-w / kT)
    Z = float(np.prod(1 + boltz))
    F = -kT * math.log(Z) if Z > 0 else float("-inf")
    P_v = boltz / (1 + boltz)
    mean_E = float(np.sum(w * P_v))
    S = (mean_E - F) / temperature if temperature > 0 else 0.0
    C = float(np.sum(w**2 * P_v * (1 - P_v))) / (kT**2)
    return PartitionResult(Z=Z, free_energy=F, mean_energy=mean_E,
                           entropy=S, specific_heat=C, violation_probabilities=P_v)


# ── Phase transitions ───────────────────────────────────────

@dataclass
class PhaseTransitionResult:
    critical_index: int
    critical_violation_rate: float
    is_transition_detected: bool
    violation_rates: np.ndarray
    second_derivative: np.ndarray


def detect_phase_transition(violation_rates: Sequence[float],
                            threshold: float = 2.0) -> PhaseTransitionResult:
    """Detect phase transition via second-derivative spike."""
    rates = np.array(violation_rates, dtype=float)
    if len(rates) < 3:
        return PhaseTransitionResult(
            critical_index=0,
            critical_violation_rate=rates[0] if len(rates) > 0 else 0.0,
            is_transition_detected=False,
            violation_rates=rates,
            second_derivative=np.array([]),
        )
    second_deriv = np.diff(np.diff(rates))
    abs_sd = np.abs(second_deriv)
    mean_sd = float(np.mean(abs_sd))
    std_sd = float(np.std(abs_sd)) if len(abs_sd) > 1 else 1.0
    spikes = np.where(abs_sd > mean_sd + threshold * std_sd)[0]
    detected = len(spikes) > 0
    idx = int(spikes[np.argmax(abs_sd[spikes])]) + 1 if detected else int(np.argmax(abs_sd)) + 1
    return PhaseTransitionResult(
        critical_index=idx,
        critical_violation_rate=float(rates[idx]) if idx < len(rates) else float(rates[-1]),
        is_transition_detected=detected,
        violation_rates=rates,
        second_derivative=second_deriv,
    )


# ── ThermoEngine (facade) ───────────────────────────────────

class ThermoEngine:
    """
    Unified thermodynamic analysis of constraint systems.

    Practical interpretation guide:
        - Z close to 1.0 → system is over-constrained (very few valid states)
        - Z large       → many possible states (system is under-constrained)
        - Temperature high → less strict checking (more values pass)
        - Temperature low  → very strict checking (only tight fits pass)
        - Entropy high → violations are scattered across many constraints
        - Entropy low  → violations are concentrated in few constraints
        - Free energy → how much useful constraint-satisfying capacity remains
        - Specific heat → how sensitive the system is to temperature changes
        - ideal_gas_check() → True means constraints are independent

    Usage:
        engine = ThermoEngine(weights=[1.0, 2.0, 0.5])
        result = engine.partition(temperature=1.0)
        phase = engine.phase_transition([0.01, 0.02, 0.05, 0.4, 0.9])
    """

    def __init__(self, weights: np.ndarray | list[float], k: float = 1.0):
        self.weights = np.asarray(weights, dtype=float)
        self.k = k
        self.n = len(self.weights)

    def partition(self, temperature: float = 1.0) -> PartitionResult:
        return partition_function(self.weights, temperature, self.k)

    def entropy(self, n_violated: int, base: float = 2.0) -> float:
        return violation_entropy(self.n, n_violated, base)

    def normalized_entropy(self, n_violated: int) -> float:
        return normalized_entropy(self.n, n_violated)

    def phase_transition(self, violation_rates: list[float],
                         threshold: float = 2.0) -> PhaseTransitionResult:
        return detect_phase_transition(violation_rates, threshold)

    def temperature(self, violation_energies: np.ndarray,
                    n_violated: int) -> float:
        """Constraint temperature T = <E> / S. Returns inf if S = 0."""
        if n_violated == 0 or n_violated == self.n:
            return float("inf")
        mean_e = float(np.mean(violation_energies)) if len(violation_energies) > 0 else 0.0
        s = violation_entropy(self.n, n_violated, math.e)
        return mean_e / s if s > 0 else float("inf")

    def ideal_gas_check(self) -> bool:
        """
        Verify ideal gas law: Z_factorized == Z_monolithic.
        For independent constraints, Z = prod(Z_i).
        """
        Z_total = float(np.prod(1 + np.exp(-self.weights / self.k)))
        Z_factors = [float(1 + math.exp(-w / self.k)) for w in self.weights]
        Z_prod = math.prod(Z_factors)
        return math.isclose(Z_total, Z_prod, rel_tol=1e-12)

    def summary(self, temperature: float = 1.0) -> dict:
        p = self.partition(temperature)
        return {
            "n_constraints": self.n,
            "Z": p.Z,
            "free_energy": p.free_energy,
            "mean_energy": p.mean_energy,
            "entropy": p.entropy,
            "specific_heat": p.specific_heat,
            "ideal_gas": self.ideal_gas_check(),
        }
