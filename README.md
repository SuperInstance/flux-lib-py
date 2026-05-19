# flux-lib

Constraint theory as a single import.

```python
from flux_lib import ConstraintEngine, fracture, coalesce, SedimentStack
from flux_lib import ShadowgapFinder, ThermoEngine
```

## Install

```bash
pip install -e .
```

## What's Inside

| Module | What it does |
|--------|-------------|
| `ConstraintEngine` | Exact bound checking, zero false negatives. NaN always violates. |
| `fracture` / `coalesce` | Split constraint systems into independent blocks. Bitwise OR merge. |
| `SedimentStack` | Accumulated edge-case corrections. Monotonic correctness guarantee. |
| `ShadowgapFinder` | Find what ALL checkers miss. Information-theoretic blind spot detection. |
| `ThermoEngine` | Partition function Z, entropy, temperature, phase transitions. |

## Quick Start

### Constraint Checking

```python
from flux_lib import ConstraintEngine

# From industry preset
eng = ConstraintEngine.from_preset("automotive_can")
result = eng.check(9000)  # engine RPM out of range
print(result.passed)       # False
print(result.severity.name) # CRITICAL

# Batch (numpy vectorized)
import numpy as np
masks = eng.check_batch(np.array([3000, 9000, -40]))
# → array([0, 1, 4], dtype=uint8)
```

### Industry Presets

10 presets built in: `automotive_can`, `aviation_adsb`, `medical_fhir`,
`financial_fix`, `energy_scada`, `iot_mqtt`, `maritime_nmea`,
`nuclear_reactor`, `railway_ertms`, `robotics`.

### Fracture & Coalesce

```python
from flux_lib import fracture, coalesce
from flux_lib.fracture import DependencyGraph

# Build dependency graph
graph = DependencyGraph.from_masks([
    np.array([0]),       # constraint 0 → dimension 0
    np.array([1]),       # constraint 1 → dimension 1 (independent)
    np.array([0, 2]),    # constraint 2 → dimensions 0,2 (coupled with c0)
])
result = fracture(graph)
print(result.n_blocks)           # 2 (c0+c2 in one block, c1 alone)
print(result.speedup_potential)  # 1.5x

# Merge results
total_mask = coalesce([0b01, 0b10])  # → 0b11
```

### Sediment Stack

```python
from flux_lib import SedimentStack, ConstraintCorrection

stack = SedimentStack()
stack.add_layer("arctic deployment", corrections=[
    ConstraintCorrection("coolant_temp_c", new_lo=-55, reason="arctic ops")
])
# Apply corrections to a check result
lo, hi, passed, n = stack.apply("coolant_temp_c", -50, -40, 150, True)
```

### Shadowgap Discovery

```python
from flux_lib import ShadowgapFinder, MultiChecker
import numpy as np

lo = np.array([0.0, 0.0, 0.0])
hi = np.array([100.0, 100.0, 100.0])
checker = MultiChecker(lo, hi)
values = np.random.uniform(-20, 120, (200, 3))

finder = ShadowgapFinder(n_constraints=3)
result = finder.find_from_checker(checker, values)
print(f"Blind spots: {result.n_shadowgap}")
```

### Thermodynamic Analysis

```python
from flux_lib import ThermoEngine

engine = ThermoEngine([1.0, 2.0, 0.5])
p = engine.partition(temperature=1.0)
print(f"Z = {p.Z}, F = {p.free_energy}, S = {p.entropy}")
print(f"Ideal gas (independent): {engine.ideal_gas_check()}")
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Core Theorems

1. **Zero false negatives**: A value outside bounds is always detected. No exceptions.
2. **Fracture-coalesce**: Independent blocks → bitwise OR merge → provably correct.
3. **Monotonic sediment**: N layers has strictly higher correctness than N-1.
4. **Shadowgap convergence**: Each correction reduces future blind spot rate.
5. **Ideal gas law**: Independent constraints → Z factorizes as product.

## License

MIT
