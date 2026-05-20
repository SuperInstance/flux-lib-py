"""Tests for serialization — save/load configs."""

import json
import os
import tempfile

import pytest
from flux_lib import ConstraintEngine


class TestToDict:
    def test_basic_to_dict(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "temp"},
            {"lo": -40, "hi": 150, "name": "coolant"},
        ])
        d = eng.to_dict()
        assert len(d["constraints"]) == 2
        assert d["constraints"][0]["name"] == "temp"
        assert d["constraints"][0]["lo"] == 0
        assert d["constraints"][0]["hi"] == 100
        assert d["constraints"][1]["name"] == "coolant"
        assert d["preset"] is None

    def test_to_dict_with_severity(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "x", "severity": 3},
        ])
        d = eng.to_dict()
        assert d["constraints"][0]["severity"] == 3

    def test_to_dict_with_preset(self):
        eng = ConstraintEngine.from_preset("automotive_can")
        d = eng.to_dict()
        assert d["preset"] == "automotive_can"
        assert len(d["constraints"]) == 8

    def test_to_dict_is_json_serializable(self):
        eng = ConstraintEngine([
            {"lo": 0.5, "hi": 99.9, "name": "sensor"},
        ])
        d = eng.to_dict()
        s = json.dumps(d)
        assert "sensor" in s


class TestFromDict:
    def test_roundtrip_basic(self):
        original = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "temp"},
            {"lo": -40, "hi": 150, "name": "coolant"},
        ])
        d = original.to_dict()
        restored = ConstraintEngine.from_dict(d)
        assert restored.n == original.n
        assert restored._names == original._names
        assert restored._lo == original._lo
        assert restored._hi == original._hi

    def test_roundtrip_preserves_checking(self):
        original = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "x"},
            {"lo": -10, "hi": 10, "name": "y"},
        ])
        d = original.to_dict()
        restored = ConstraintEngine.from_dict(d)
        # Same checking behavior
        r1 = original.check_vector([50, 5])
        r2 = restored.check_vector([50, 5])
        assert r1.passed == r2.passed
        assert r1.error_mask == r2.error_mask

        r1 = original.check_vector([200, -20])
        r2 = restored.check_vector([200, -20])
        assert r1.error_mask == r2.error_mask == 0b11

    def test_from_dict_with_preset_name(self):
        d = {
            "constraints": [{"lo": 0, "hi": 100, "name": "x"}],
            "preset": "custom_preset",
        }
        eng = ConstraintEngine.from_dict(d)
        assert eng._preset_name == "custom_preset"

    def test_from_dict_missing_preset(self):
        d = {"constraints": [{"lo": 0, "hi": 100, "name": "x"}]}
        eng = ConstraintEngine.from_dict(d)
        assert getattr(eng, "_preset_name", None) is None


class TestSaveLoad:
    def test_save_and_load_file(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "temp"},
            {"lo": -40, "hi": 150, "name": "coolant"},
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            eng.save(path)
            assert os.path.exists(path)

            restored = ConstraintEngine.load(path)
            assert restored.n == eng.n
            assert restored._names == eng._names
            assert restored._lo == eng._lo
            assert restored._hi == eng._hi
        finally:
            os.unlink(path)

    def test_save_produces_valid_json(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 100, "name": "x", "severity": 3},
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            eng.save(path)
            with open(path) as f:
                data = json.load(f)
            assert data["constraints"][0]["severity"] == 3
        finally:
            os.unlink(path)

    def test_save_load_preset_engine(self):
        eng = ConstraintEngine.from_preset("automotive_can")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            eng.save(path)
            restored = ConstraintEngine.load(path)
            assert restored.n == 8
            assert restored._preset_name == "automotive_can"
        finally:
            os.unlink(path)

    def test_roundtrip_full_checking(self):
        eng = ConstraintEngine([
            {"lo": 0, "hi": 8000, "name": "rpm"},
            {"lo": -40, "hi": 150, "name": "temp"},
            {"lo": 0, "hi": 100, "name": "fuel"},
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            eng.save(path)
            restored = ConstraintEngine.load(path)
            # Exact same checking
            for vals in [[3000, 90, 50], [99999, 90, 50], [3000, 999, 50]]:
                r1 = eng.check_vector(vals)
                r2 = restored.check_vector(vals)
                assert r1.error_mask == r2.error_mask
                assert r1.passed == r2.passed
        finally:
            os.unlink(path)
