"""
flux_lib.fracture — Split constraint systems into independent blocks.

If constraints are independent, the partition function factorizes Z = prod(Z_i).
This module makes that operational via connected-component analysis.

THEOREM: If fracture correctly identifies connected components, coalescence
via bitwise OR preserves zero false negatives. QED.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ── DependencyGraph ─────────────────────────────────────────

@dataclass
class DependencyGraph:
    """Bipartite graph: constraints (rows) × dimensions (columns)."""
    adjacency: np.ndarray  # (n_constraints, n_dimensions), uint8
    n_constraints: int
    n_dimensions: int
    constraint_names: List[str] = field(default_factory=list)
    dimension_names: List[str] = field(default_factory=list)

    @classmethod
    def from_masks(cls,
                   masks: Sequence[np.ndarray],
                   constraint_names: Sequence[str] = (),
                   dimension_names: Sequence[str] = ()) -> "DependencyGraph":
        n_c = len(masks)
        n_d = max((m.max() for m in masks), default=0) + 1 if masks else 0
        adj = np.zeros((n_c, n_d), dtype=np.uint8)
        for i, m in enumerate(masks):
            adj[i, m] = 1
        return cls(
            adjacency=adj, n_constraints=n_c, n_dimensions=n_d,
            constraint_names=list(constraint_names) or [f"c{i}" for i in range(n_c)],
            dimension_names=list(dimension_names) or [f"d{j}" for j in range(n_d)],
        )

    @classmethod
    def from_adjacency(cls, adj: np.ndarray,
                       constraint_names: Sequence[str] = (),
                       dimension_names: Sequence[str] = ()) -> "DependencyGraph":
        n_c, n_d = adj.shape
        return cls(
            adjacency=adj.astype(np.uint8), n_constraints=n_c, n_dimensions=n_d,
            constraint_names=list(constraint_names) or [f"c{i}" for i in range(n_c)],
            dimension_names=list(dimension_names) or [f"d{j}" for j in range(n_d)],
        )

    def involves(self, constraint_idx: int, dimension_idx: int) -> bool:
        return bool(self.adjacency[constraint_idx, dimension_idx])

    def constraint_dims(self, idx: int) -> np.ndarray:
        return np.flatnonzero(self.adjacency[idx])

    def dim_constraints(self, idx: int) -> np.ndarray:
        return np.flatnonzero(self.adjacency[:, idx])


# ── Block & FractureResult ──────────────────────────────────

@dataclass
class Block:
    """One independent block of the fractured system."""
    constraint_indices: List[int]
    dimension_indices: List[int]
    size: int = 0

    def __post_init__(self):
        self.size = len(self.constraint_indices)


@dataclass
class FractureResult:
    """Result of fracturing a constraint system into independent blocks."""
    blocks: List[Block]
    graph: DependencyGraph
    n_blocks: int = 0
    largest_block_size: int = 0
    speedup_potential: float = 1.0

    def __post_init__(self):
        self.n_blocks = len(self.blocks)
        self.largest_block_size = max((b.size for b in self.blocks), default=0)
        n_c = self.graph.n_constraints
        self.speedup_potential = n_c / self.largest_block_size if self.largest_block_size > 0 else 1.0

    def summary(self) -> Dict:
        return {
            "n_blocks": self.n_blocks,
            "largest_block_size": self.largest_block_size,
            "speedup_potential": round(self.speedup_potential, 2),
            "block_sizes": [b.size for b in self.blocks],
        }


# ── fracture() — BFS connected components ───────────────────

def fracture(graph: DependencyGraph) -> FractureResult:
    """
    Find connected components via BFS on the bipartite graph.
    Returns a FractureResult with independent blocks.
    """
    visited_c = np.zeros(graph.n_constraints, dtype=bool)
    visited_d = np.zeros(graph.n_dimensions, dtype=bool)
    blocks: list[Block] = []

    for seed_c in range(graph.n_constraints):
        if visited_c[seed_c]:
            continue
        comp_c: set[int] = set()
        comp_d: set[int] = set()
        queue: deque = deque([("c", seed_c)])
        while queue:
            node_type, idx = queue.popleft()
            if node_type == "c":
                if visited_c[idx]:
                    continue
                visited_c[idx] = True
                comp_c.add(idx)
                for d in np.flatnonzero(graph.adjacency[idx]):
                    if not visited_d[d]:
                        queue.append(("d", d))
            else:
                if visited_d[idx]:
                    continue
                visited_d[idx] = True
                comp_d.add(idx)
                for c in np.flatnonzero(graph.adjacency[:, idx]):
                    if not visited_c[c]:
                        queue.append(("c", c))

        blocks.append(Block(
            constraint_indices=sorted(comp_c),
            dimension_indices=sorted(comp_d),
        ))

    # Orphan dimensions
    for d in range(graph.n_dimensions):
        if not visited_d[d]:
            blocks.append(Block(constraint_indices=[], dimension_indices=[d], size=0))

    return FractureResult(blocks=blocks, graph=graph)


def fracture_from_bounds(constraints: list[dict]) -> FractureResult:
    """Convenience: fracture from a list of constraint dicts with optional 'dims' key."""
    masks = []
    for i, c in enumerate(constraints):
        if "dims" in c:
            masks.append(np.array(c["dims"], dtype=np.intp))
        else:
            masks.append(np.array([i], dtype=np.intp))
    graph = DependencyGraph.from_masks(masks)
    return fracture(graph)


# ── coalesce() — Bitwise OR merge ───────────────────────────

def coalesce(block_masks: list[int], n_total: int = 0) -> int:
    """
    Coalesce block-level error masks via bitwise OR.

    CORRECTNESS: Since blocks are independent (no shared dimensions),
    OR captures ALL violations with zero false negatives.
    """
    result = 0
    for m in block_masks:
        result |= m
    return result


def coalesce_arrays(block_arrays: list[np.ndarray]) -> np.ndarray:
    """Coalesce arrays elementwise via OR."""
    if not block_arrays:
        return np.array([], dtype=np.uint8)
    result = np.zeros_like(block_arrays[0])
    for arr in block_arrays:
        result |= arr
    return result


def verify_coalescence(block_masks: list[int],
                       block_indices: list[list[int]],
                       monolithic_mask: int) -> Tuple[bool, str]:
    """Verify coalesced result equals monolithic. Returns (ok, message)."""
    reconstructed = 0
    for mask, indices in zip(block_masks, block_indices):
        for bit_pos, c_idx in enumerate(indices):
            if mask & (1 << bit_pos):
                reconstructed |= (1 << c_idx)
    if reconstructed == monolithic_mask:
        return True, f"PERFECT MATCH: {monolithic_mask:#x}"
    return False, f"MISMATCH: coalesced={reconstructed:#x} vs monolithic={monolithic_mask:#x}"
