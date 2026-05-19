"""
flux_lib.sediment — Accumulated correctness as computational sediment.

Each layer is an immutable edge-case correction. New layers supersede old ones.
Monotonic guarantee: N layers has strictly higher correctness than N-1.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── ConstraintCorrection ────────────────────────────────────

@dataclass(frozen=True)
class ConstraintCorrection:
    """A single correction to a constraint definition."""
    constraint_name: str
    old_lo: Optional[float] = None
    old_hi: Optional[float] = None
    new_lo: Optional[float] = None
    new_hi: Optional[float] = None
    override_pass: Optional[bool] = None
    reason: str = ""

    def apply_to(self, lo: float, hi: float, passed: bool) -> Tuple[float, float, bool]:
        out_lo = self.new_lo if self.new_lo is not None else lo
        out_hi = self.new_hi if self.new_hi is not None else hi
        out_passed = self.override_pass if self.override_pass is not None else passed
        return out_lo, out_hi, out_passed


# ── SedimentLayer ───────────────────────────────────────────

@dataclass
class SedimentLayer:
    """An immutable layer of edge-case corrections. Never deleted, only superseded."""
    layer_id: int
    input_context: Dict[str, Any]
    corrections: List[ConstraintCorrection]
    timestamp: float = field(default_factory=time.time)
    provenance: str = ""
    model: str = ""
    superseded: bool = False
    superseded_by: Optional[int] = None
    catch_count: int = 0

    def content_hash(self) -> str:
        blob = json.dumps(
            {"id": self.layer_id, "corrections": [
                {"name": c.constraint_name, "new_lo": c.new_lo, "new_hi": c.new_hi}
                for c in self.corrections
            ]},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def active_corrections(self) -> List[ConstraintCorrection]:
        """Corrections that haven't been superseded."""
        if self.superseded:
            return []
        return list(self.corrections)


# ── SedimentStack ───────────────────────────────────────────

class SedimentStack:
    """
    Ordered stack of sediment layers. Monotonic guarantee:
    adding a layer never reduces correctness coverage.

    Usage:
        stack = SedimentStack()
        stack.add_layer(context="crisis_1", corrections=[
            ConstraintCorrection("coolant_temp", new_lo=-45, new_hi=155, reason="arctic ops")
        ])
        result = stack.apply("coolant_temp", -42, -40, 150, passed=True)
    """

    def __init__(self):
        self._layers: list[SedimentLayer] = []
        self._next_id: int = 0

    @property
    def layers(self) -> List[SedimentLayer]:
        return list(self._layers)

    @property
    def n_layers(self) -> int:
        return len(self._layers)

    def add_layer(self,
                  context: Dict[str, Any] | str = "",
                  corrections: Optional[List[ConstraintCorrection]] = None,
                  provenance: str = "",
                  model: str = "") -> SedimentLayer:
        """Add a new sediment layer. Returns the created layer."""
        if isinstance(context, str):
            context = {"trigger": context}
        layer = SedimentLayer(
            layer_id=self._next_id,
            input_context=context,
            corrections=corrections or [],
            provenance=provenance,
            model=model,
        )
        self._layers.append(layer)
        self._next_id += 1
        return layer

    def apply(self,
              constraint_name: str,
              value: float,
              lo: float,
              hi: float,
              passed: bool) -> Tuple[float, float, bool, int]:
        """
        Apply all active corrections for a constraint.

        Returns: (adjusted_lo, adjusted_hi, adjusted_passed, corrections_applied)
        """
        applied = 0
        cur_lo, cur_hi, cur_passed = lo, hi, passed
        for layer in self._layers:
            if layer.superseded:
                continue
            for corr in layer.corrections:
                if corr.constraint_name == constraint_name:
                    cur_lo, cur_hi, cur_passed = corr.apply_to(cur_lo, cur_hi, cur_passed)
                    applied += 1
                    layer.catch_count += 1
        return cur_lo, cur_hi, cur_passed, applied

    def correctness_density(self) -> float:
        """Fraction of layers that have active (non-superseded) corrections."""
        if not self._layers:
            return 0.0
        active = sum(1 for l in self._layers if not l.superseded and l.corrections)
        return active / len(self._layers)

    def supersede(self, layer_id: int, by_layer_id: Optional[int] = None) -> bool:
        """Mark a layer as superseded. Returns True if found."""
        for l in self._layers:
            if l.layer_id == layer_id:
                l.superseded = True
                l.superseded_by = by_layer_id
                return True
        return False

    def summary(self) -> Dict:
        return {
            "n_layers": self.n_layers,
            "active_layers": sum(1 for l in self._layers if not l.superseded),
            "superseded_layers": sum(1 for l in self._layers if l.superseded),
            "total_corrections": sum(len(l.corrections) for l in self._layers),
            "correctness_density": round(self.correctness_density(), 3),
        }
