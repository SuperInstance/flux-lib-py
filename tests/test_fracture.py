"""Tests for flux_lib.fracture — dependency graph, fracture, coalesce."""

import numpy as np
import pytest
from flux_lib.fracture import (
    DependencyGraph, Block, FractureResult,
    fracture, fracture_from_bounds, coalesce, coalesce_arrays, verify_coalescence,
)


class TestDependencyGraph:
    def test_from_masks_basic(self):
        masks = [np.array([0]), np.array([1]), np.array([0, 2])]
        g = DependencyGraph.from_masks(masks)
        assert g.n_constraints == 3
        assert g.n_dimensions == 3

    def test_involves(self):
        masks = [np.array([0, 1]), np.array([2])]
        g = DependencyGraph.from_masks(masks)
        assert g.involves(0, 0)
        assert g.involves(0, 1)
        assert not g.involves(0, 2)
        assert g.involves(1, 2)

    def test_from_adjacency(self):
        adj = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        g = DependencyGraph.from_adjacency(adj)
        assert g.n_constraints == 2
        assert g.n_dimensions == 2

    def test_constraint_dims(self):
        masks = [np.array([0, 2, 4])]
        g = DependencyGraph.from_masks(masks)
        dims = g.constraint_dims(0)
        np.testing.assert_array_equal(dims, [0, 2, 4])


class TestFracture:
    def test_independent_constraints(self):
        # 3 constraints, each on different dimension
        masks = [np.array([0]), np.array([1]), np.array([2])]
        g = DependencyGraph.from_masks(masks)
        result = fracture(g)
        assert result.n_blocks == 3
        assert result.speedup_potential == 3.0

    def test_coupled_constraints(self):
        # 2 constraints sharing dimension 0
        masks = [np.array([0]), np.array([0, 1])]
        g = DependencyGraph.from_masks(masks)
        result = fracture(g)
        assert result.n_blocks == 1
        assert result.largest_block_size == 2

    def test_mixed_coupling(self):
        # c0 -> d0, c1 -> d1, c2 -> d0, d2
        # c0 and c2 share d0 → same block; c1 is independent
        masks = [np.array([0]), np.array([1]), np.array([0, 2])]
        g = DependencyGraph.from_masks(masks)
        result = fracture(g)
        assert result.n_blocks == 2

    def test_single_constraint(self):
        masks = [np.array([0])]
        g = DependencyGraph.from_masks(masks)
        result = fracture(g)
        assert result.n_blocks == 1

    def test_fracture_from_bounds(self):
        constraints = [
            {"lo": 0, "hi": 100, "name": "a"},
            {"lo": -10, "hi": 10, "name": "b"},
        ]
        result = fracture_from_bounds(constraints)
        assert result.n_blocks == 2


class TestCoalesce:
    def test_coalesce_masks(self):
        assert coalesce([0b01, 0b10]) == 0b11
        assert coalesce([0b00, 0b00]) == 0b00
        assert coalesce([0b11, 0b01]) == 0b11

    def test_coalesce_arrays(self):
        a = np.array([0b01, 0b00], dtype=np.uint8)
        b = np.array([0b10, 0b01], dtype=np.uint8)
        result = coalesce_arrays([a, b])
        np.testing.assert_array_equal(result, [0b11, 0b01])

    def test_coalesce_empty(self):
        result = coalesce_arrays([])
        assert len(result) == 0

    def test_verify_coalescence_match(self):
        ok, msg = verify_coalescence([0b01], [[0]], 0b01)
        assert ok

    def test_verify_coalescence_mismatch(self):
        ok, msg = verify_coalescence([0b00], [[0]], 0b01)
        assert not ok


class TestFractureResult:
    def test_summary(self):
        masks = [np.array([0]), np.array([1])]
        g = DependencyGraph.from_masks(masks)
        result = fracture(g)
        s = result.summary()
        assert "n_blocks" in s
        assert "speedup_potential" in s
        assert s["n_blocks"] == 2
