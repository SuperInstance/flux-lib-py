"""
flux_lib — Constraint theory in one import.

    from flux_lib import ConstraintEngine, fracture, coalesce, SedimentStack
    from flux_lib import ShadowgapFinder, ThermoEngine

"""

from flux_lib.core import ConstraintEngine, CheckResult, Severity
from flux_lib.fracture import DependencyGraph, fracture, coalesce, FractureResult
from flux_lib.sediment import SedimentLayer, SedimentStack, ConstraintCorrection
from flux_lib.shadowgap import ShadowgapFinder, ShadowgapResult, MultiChecker
from flux_lib.drift import DriftDetector
from flux_lib.thermo import ThermoEngine

__all__ = [
    "ConstraintEngine", "CheckResult", "Severity",
    "DependencyGraph", "fracture", "coalesce", "FractureResult",
    "SedimentLayer", "SedimentStack", "ConstraintCorrection",
    "ShadowgapFinder", "ShadowgapResult", "MultiChecker",
    "ThermoEngine",
    "DriftDetector",
]
__version__ = "0.1.0"
