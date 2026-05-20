# flux-lib

Constraint theory as a single import:

```python
from flux_lib import ConstraintEngine, fracture, coalesce, SedimentStack
from flux_lib import ShadowgapFinder, ThermoEngine, DriftDetector
```

```bash
pip install -e .
```

## How It Works

At its core, constraint theory asks: *given a set of bounds, does every value fall within spec?* The answer is a bitmask — one bit per constraint, zero means pass. Everything else in the library builds on that foundation.

There are two ideas that go deeper than simple bounds checking:

### Thermodynamics: Temperature as Strictness

The `ThermoEngine` maps constraint systems onto statistical mechanics. Each constraint is like an energy level, and values are particles:

- **Temperature** controls how strict the system is. Low temperature → only exact matches pass. High temperature → the system tolerates more deviation.
- **Entropy** measures how "spread out" the constraint violations are. High entropy means violations are scattered across many constraints. Low entropy means they're concentrated.
- **Free energy** (F = -T ln Z) tells you the useful constraint-satisfying capacity of the system.
- **Partition function** Z sums over all possible states. When constraints are independent, Z factorizes — this is the *ideal gas law* for constraints.

This isn't just an analogy. The math is identical to statistical mechanics, and it gives you real diagnostic tools: phase transitions tell you when a system fundamentally changes behavior.

### Shadowgap: What All Checkers Miss

A *shadowgap* is a region of input space where no checker in your ensemble detects a violation — a blind spot shared by every checker you have. The `ShadowgapFinder` uses information-theoretic analysis to find these gaps.

Think of it this way: if you have three checkers and none of them catch a particular kind of drift, that drift is invisible to your entire system. Shadowgap detection finds those invisible regions before they cause problems.

## What This Module Does

It's the unified library that pulls together all the flux subsystems: exact checking, fracture-coalesce for parallelization, sediment layers for edge-case corrections, shadowgap detection for blind spots, and thermodynamic analysis for system-level diagnostics.

10 industry presets built in: `automotive_can`, `aviation_adsb`, `medical_fhir`, `financial_fix`, `energy_scada`, `iot_mqtt`, `maritime_nmea`, `nuclear_reactor`, `railway_ertms`, `robotics`.

## Constraint Checking

**The error mask is uint8, supporting up to 8 constraints.** For more constraints, use multiple engines.

```python
from flux_lib import ConstraintEngine

eng = ConstraintEngine.from_preset("automotive_can")
result = eng.check(9000)  # checks one value against ALL 8 constraints
print(result.passed)       # False
print(result.severity.name) # CRITICAL

# Batch (numpy vectorized)
import numpy as np
masks = eng.check_batch(np.array([3000, 9000, -40]))
# → array([0, 1, 4], dtype=uint8)
```

### check_vector — N values against N constraints

`check()` tests ONE value against ALL constraints. For sensor arrays where each reading
has its own bounds, use `check_vector`:

```python
eng = ConstraintEngine.from_preset("automotive_can")
# 8 sensor readings, one per constraint
result = eng.check_vector([3000, 65, 90, 40, 10, 45, 12.5, 50])
print(result.passed)        # True — each value checked against its OWN constraint
print(result.error_mask)    # 0

result = eng.check_vector([3000, 65, 90, 40, 10, 45, 12.5, 999])
print(result.passed)        # False — fuel out of range
print(result.violated_count) # 1

# Batch mode for time-series
samples = np.array([
    [3000, 65, 90, 40, 10, 45, 12.5, 50],
    [9000, 65, 90, 40, 10, 45, 12.5, 50],  # RPM violation
])
masks = eng.check_vector_batch(samples)  # → array([0, 1], dtype=uint8)
```

## Fracture & Coalesce

Split independent constraint blocks for parallel checking, then merge:

```python
from flux_lib import fracture, coalesce
from flux_lib.fracture import DependencyGraph

graph = DependencyGraph.from_masks([
    np.array([0]),       # constraint 0 → dimension 0
    np.array([1]),       # constraint 1 → dimension 1 (independent)
    np.array([0, 2]),    # constraint 2 → dimensions 0,2 (coupled with c0)
])
result = fracture(graph)
print(result.n_blocks)           # 2
print(result.speedup_potential)  # 1.5x

total_mask = coalesce([0b01, 0b10])  # → 0b11
```

## Sediment Stack

Immutable correction layers:

```python
from flux_lib import SedimentStack, ConstraintCorrection

stack = SedimentStack()
stack.add_layer("arctic deployment", corrections=[
    ConstraintCorrection("coolant_temp_c", new_lo=-55, reason="arctic ops")
])
lo, hi, passed, n = stack.apply("coolant_temp_c", -50, -40, 150, True)
```

## Shadowgap Discovery

Find blind spots shared by all checkers:

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

## Thermodynamic Analysis

```python
from flux_lib import ThermoEngine

engine = ThermoEngine([1.0, 2.0, 0.5])
p = engine.partition(temperature=1.0)
print(f"Z = {p.Z}, F = {p.free_energy}, S = {p.entropy}")
print(f"Independent constraints: {engine.ideal_gas_check()}")
```

### Practical Interpretation

| Value | Meaning |
|-------|---------|
| **Z close to 1.0** | System is over-constrained — very few valid states |
| **Z large** | System is under-constrained — many possible states |
| **Temperature high** | Less strict checking (more values pass) |
| **Temperature low** | Very strict checking (only tight fits pass) |
| **Entropy high** | Violations are scattered across many constraints |
| **Entropy low** | Violations are concentrated in few constraints |
| **Free energy** | How much useful constraint-satisfying capacity remains |
| **Specific heat** | How sensitive the system is to temperature changes |
| **ideal_gas_check() = True** | Constraints are independent (no coupling) |

## Performance

Exact checking is zero-allocation on the hot path. Batch mode uses numpy vectorization. Fracture-coalesce speedup depends on your constraint graph's independence structure. Run your own benchmarks with `pytest --benchmark` or time the batch operations directly.

## Where to Go Next

| If you want... | Go to |
|----------------|-------|
| CLI tool for quick checks | [flux-check](../flux-check-py) |
| Hyperbolic model routing | [flux-hyperbolic](../flux-hyperbolic-py) |
| Genetic expression engine | [flux-genome](../flux-genome-py) |

## Serialization — Save/Load Configs

```python
eng = ConstraintEngine.from_preset("automotive_can")

# Export to dict (JSON-serializable)
config = eng.to_dict()

# Save to file
eng.save("automotive_config.json")

# Load from file
restored = ConstraintEngine.load("automotive_config.json")
assert restored.n == eng.n  # identical checking behavior

# Round-trip via dict
restored2 = ConstraintEngine.from_dict(config)
```

## Aggregation — Batch Analysis

```python
eng = ConstraintEngine.from_preset("automotive_can")
readings = [
    [3000, 65, 90, 40, 10, 45, 12.5, 50],   # all pass
    [9000, 65, 90, 40, 10, 45, 12.5, 50],   # RPM violation
    [3000, 65, 90, 40, 10, 45, 7.5, 50],    # battery violation
]
result = eng.check_and_aggregate(readings)
print(result["violation_rate"])                    # 0.083 (2/24)
print(result["per_constraint_violation_rate"])     # per-sensor rates
print(result["worst_reading"])                     # (index, CheckResult)
print(result["severity_breakdown"])                # {"PASS": 1, "WARNING": 2, ...}
```

## Drift Detection

```python
from flux_lib import DriftDetector

det = DriftDetector(window_size=100)
bounds = [(0, 8000), (-40, 150), (0, 100)]  # lo/hi per sensor

for reading in sensor_stream:
    det.add(reading)
    if det.n >= 20:
        drift = det.detect_drift(bounds=bounds)
        if drift["drifting"]:
            for name, info in drift["per_sensor"].items():
                print(f"{name}: {info['direction']} at rate {info['rate']:.3f}")
            # Time-to-violation estimates
            for name, ttv in drift["time_to_violation"].items():
                if ttv is not None:
                    print(f"  {name}: ~{ttv:.0f} readings until violation")

# Forecast next 10 readings based on current trend
forecasts = det.forecast(n_ahead=10)
```

## Thermodynamic Recommendations

```python
from flux_lib import ThermoEngine

engine = ThermoEngine([1.0, 2.0, 0.5])
rec = engine.recommend(temperature=1.0)
print(rec["action"])              # 'maintain' | 'tighten' | 'loosen' | 'investigate'
print(rec["reason"])             # human-readable explanation
print(rec["suggested_temperature"])  # adjustment if needed
print(rec["focus_constraints"])     # indices of constraints to focus on
```

| Action | Meaning | When |
|--------|---------|------|
| `tighten` | Narrow bounds or lower temperature | Z > 100 (under-constrained) |
| `loosen` | Widen bounds or raise temperature | Z < 1.5 (over-constrained) |
| `investigate` | Constraints are coupled — use fracture() | Over-constrained + not ideal gas |
| `maintain` | System is operating normally | Z in healthy range |

## Core Properties

1. **Zero false negatives**: A value outside bounds is always detected.
2. **Fracture-coalesce**: Independent blocks → bitwise OR merge → provably correct.
3. **Monotonic sediment**: N layers has strictly higher correctness than N-1.
4. **Shadowgap convergence**: Each correction reduces future blind spot rate.
5. **Ideal gas law**: Independent constraints → partition function factorizes.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
