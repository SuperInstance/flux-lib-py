#!/usr/bin/env python3
"""
flux-lib demo — 5-minute walkthrough of the constraint-theory library.
"""

import numpy as np
from flux_lib import (
    ConstraintEngine, fracture, coalesce, SedimentStack,
    ConstraintCorrection, ShadowgapFinder, MultiChecker, ThermoEngine,
)

print("═" * 60)
print("flux-lib demo — Constraint Theory in One Import")
print("═" * 60)

# 1. ConstraintEngine
print("\n1. ConstraintEngine — exact bound checking")
print("-" * 40)
eng = ConstraintEngine.from_preset("automotive_can")
print(f"   Engine: {eng}")
print(f"   Constraints: {[c.name for c in eng.constraints]}")

r = eng.check(50)
print(f"   check(50)    → passed={r.passed}, mask={r.error_mask}")

r = eng.check(9000)
print(f"   check(9000)  → passed={r.passed}, severity={r.severity.name}")

masks = eng.check_batch(np.array([50, 9000, -40, 151]))
print(f"   batch:       → {masks}")

rate = eng.benchmark(100_000)
print(f"   benchmark:   → {rate/1e6:.1f}M checks/sec")

# 2. Fracture & Coalesce
print("\n2. Fracture — split into independent blocks")
print("-" * 40)
from flux_lib.fracture import DependencyGraph
masks_in = [np.array([0]), np.array([1]), np.array([0, 2])]
graph = DependencyGraph.from_masks(masks_in, constraint_names=["rpm", "speed", "temp"])
result = fracture(graph)
print(f"   Blocks: {result.n_blocks}")
for i, b in enumerate(result.blocks):
    print(f"   Block {i}: constraints={b.constraint_indices}, dims={b.dimension_indices}")
print(f"   Speedup: {result.speedup_potential:.1f}x")

# Coalesce
merged = coalesce([0b01, 0b10])
print(f"   coalesce([01, 10]) = {merged:#04x}")

# 3. SedimentStack
print("\n3. SedimentStack — accumulated correctness")
print("-" * 40)
stack = SedimentStack()
stack.add_layer("arctic ops", corrections=[
    ConstraintCorrection("coolant_temp_c", new_lo=-55, reason="arctic deployment")
])
stack.add_layer("desert ops", corrections=[
    ConstraintCorrection("coolant_temp_c", new_hi=165, reason="desert deployment")
])
lo, hi, passed, n = stack.apply("coolant_temp_c", -50, -40, 150, True)
print(f"   After sediment: lo={lo}, hi={hi}, corrections_applied={n}")
print(f"   Summary: {stack.summary()}")

# 4. Shadowgap
print("\n4. Shadowgap — finding blind spots")
print("-" * 40)
np.random.seed(42)
lo_arr = np.array([0.0, 0.0, 0.0])
hi_arr = np.array([100.0, 100.0, 100.0])
mc = MultiChecker(lo_arr, hi_arr)
values = np.random.uniform(-20, 120, (200, 3))
finder = ShadowgapFinder(n_constraints=3)
sg = finder.find_from_checker(mc, values)
print(f"   Points: {sg.n_points}")
print(f"   True violations: {sg.n_true_violations}")
print(f"   Shadowgaps: {sg.n_shadowgap}")
print(f"   Shadowgap rate: {sg.shadowgap_rate:.3f}")

# 5. ThermoEngine
print("\n5. ThermoEngine — thermodynamics of constraints")
print("-" * 40)
thermo = ThermoEngine([1.0, 2.0, 0.5, 1.5])
p = thermo.partition(temperature=1.0)
print(f"   Z = {p.Z:.4f}")
print(f"   Free energy F = {p.free_energy:.4f}")
print(f"   Entropy S = {p.entropy:.4f}")
print(f"   Specific heat C = {p.specific_heat:.4f}")
print(f"   Violation probs = {p.violation_probabilities}")
print(f"   Ideal gas: {thermo.ideal_gas_check()}")

phase = thermo.phase_transition([0.01, 0.02, 0.03, 0.05, 0.5, 0.9])
print(f"   Phase transition at index {phase.critical_index}: detected={phase.is_transition_detected}")

print("\n" + "═" * 60)
print("Done. from flux_lib import * — constraint theory, delivered.")
print("═" * 60)
