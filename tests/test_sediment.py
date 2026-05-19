"""Tests for flux_lib.sediment — accumulated correctness layers."""

import pytest
from flux_lib.sediment import SedimentStack, SedimentLayer, ConstraintCorrection


class TestConstraintCorrection:
    def test_apply_bounds_change(self):
        c = ConstraintCorrection("temp", new_lo=-50, new_hi=200)
        lo, hi, passed = c.apply_to(-40, 150, True)
        assert lo == -50
        assert hi == 200

    def test_apply_override_pass(self):
        c = ConstraintCorrection("temp", override_pass=False)
        _, _, p = c.apply_to(0, 100, True)
        assert p is False

    def test_apply_no_change(self):
        c = ConstraintCorrection("temp")  # no modifications
        lo, hi, p = c.apply_to(0, 100, True)
        assert lo == 0 and hi == 100 and p is True


class TestSedimentLayer:
    def test_content_hash_deterministic(self):
        layer = SedimentLayer(
            layer_id=1,
            input_context={"trigger": "test"},
            corrections=[ConstraintCorrection("x", new_lo=-10)],
        )
        h1 = layer.content_hash()
        h2 = layer.content_hash()
        assert h1 == h2
        assert len(h1) == 16

    def test_active_corrections(self):
        layer = SedimentLayer(
            layer_id=1,
            input_context={},
            corrections=[ConstraintCorrection("x", new_lo=-10)],
        )
        assert len(layer.active_corrections) == 1

    def test_superseded_has_no_active(self):
        layer = SedimentLayer(
            layer_id=1, input_context={},
            corrections=[ConstraintCorrection("x")],
            superseded=True,
        )
        assert len(layer.active_corrections) == 0


class TestSedimentStack:
    def test_add_layer(self):
        stack = SedimentStack()
        layer = stack.add_layer(context="test", corrections=[
            ConstraintCorrection("temp", new_lo=-50)
        ])
        assert layer.layer_id == 0
        assert stack.n_layers == 1

    def test_add_multiple_layers(self):
        stack = SedimentStack()
        stack.add_layer("first")
        stack.add_layer("second")
        assert stack.n_layers == 2

    def test_apply_correction(self):
        stack = SedimentStack()
        stack.add_layer("crisis", corrections=[
            ConstraintCorrection("coolant", new_lo=-45, new_hi=155)
        ])
        lo, hi, passed, n = stack.apply("coolant", -42, -40, 150, True)
        assert lo == -45
        assert hi == 155
        assert n == 1

    def test_apply_no_matching_correction(self):
        stack = SedimentStack()
        stack.add_layer("crisis", corrections=[
            ConstraintCorrection("other", new_lo=-50)
        ])
        lo, hi, passed, n = stack.apply("coolant", -42, -40, 150, True)
        assert lo == -40 and hi == 150 and n == 0

    def test_correctness_density_empty(self):
        stack = SedimentStack()
        assert stack.correctness_density() == 0.0

    def test_correctness_density(self):
        stack = SedimentStack()
        stack.add_layer("a", corrections=[ConstraintCorrection("x")])
        stack.add_layer("b", corrections=[])  # no corrections
        d = stack.correctness_density()
        assert 0.0 < d <= 1.0

    def test_supersede(self):
        stack = SedimentStack()
        l1 = stack.add_layer("old", corrections=[ConstraintCorrection("x")])
        stack.add_layer("new", corrections=[ConstraintCorrection("x", new_lo=-99)])
        assert stack.supersede(l1.layer_id, by_layer_id=1)
        assert l1.superseded

    def test_supersede_not_found(self):
        stack = SedimentStack()
        assert not stack.supersede(999)

    def test_summary(self):
        stack = SedimentStack()
        stack.add_layer("a", corrections=[ConstraintCorrection("x")])
        s = stack.summary()
        assert s["n_layers"] == 1
        assert "correctness_density" in s

    def test_layers_property(self):
        stack = SedimentStack()
        stack.add_layer("a")
        stack.add_layer("b")
        assert len(stack.layers) == 2
