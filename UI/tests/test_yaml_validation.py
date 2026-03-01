"""Tests for viewer.yaml validation."""
import pytest
from pathlib import Path
import tempfile
import yaml

from biovision_napari.io.yaml_schema import (
    ViewerConfig_Root,
    load_viewer_yaml,
    save_viewer_yaml,
    apply_patch,
)


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "viewer.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    return p


class TestValidYAML:
    def test_minimal_valid(self, tmp_path):
        """An empty YAML should deserialise with all defaults."""
        p = _write_yaml(tmp_path, {})
        cfg = load_viewer_yaml(p)
        assert cfg.viewer.axis_order == "TZYX"
        assert len(cfg.label_layers) >= 1

    def test_full_example(self, tmp_path):
        data = {
            "project": {"name": "test_proj"},
            "paths": {"dataset_root": "./data", "masks": "./masks"},
            "viewer": {"axis_order": "TCZYX"},
            "label_layers": [
                {"name": "cells", "color": "#ff0000", "classes": ["background", "cell"]},
            ],
            "llm": {"provider": "openai", "model": "gpt-4o", "api_key_env": "OPENAI_API_KEY"},
        }
        cfg = load_viewer_yaml(_write_yaml(tmp_path, data))
        assert cfg.project.name == "test_proj"
        assert cfg.viewer.axis_order == "TCZYX"
        assert cfg.label_layers[0].name == "cells"
        assert cfg.llm.provider == "openai"

    def test_axis_order_case_insensitive(self, tmp_path):
        data = {"viewer": {"axis_order": "tzyx"}}
        cfg = load_viewer_yaml(_write_yaml(tmp_path, data))
        assert cfg.viewer.axis_order == "TZYX"


class TestInvalidYAML:
    def test_bad_axis_order_raises(self, tmp_path):
        data = {"viewer": {"axis_order": "TXQR"}}  # Q and R are invalid
        with pytest.raises(Exception):
            load_viewer_yaml(_write_yaml(tmp_path, data))

    def test_duplicate_axis_raises(self, tmp_path):
        data = {"viewer": {"axis_order": "TTZYX"}}
        with pytest.raises(Exception):
            load_viewer_yaml(_write_yaml(tmp_path, data))

    def test_empty_label_layers_raises(self, tmp_path):
        data = {"label_layers": []}
        with pytest.raises(Exception):
            load_viewer_yaml(_write_yaml(tmp_path, data))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_viewer_yaml("/nonexistent/path/viewer.yaml")


class TestRoundtrip:
    def test_save_and_reload(self, tmp_path):
        p = _write_yaml(tmp_path, {"project": {"name": "roundtrip_test"}})
        cfg = load_viewer_yaml(p)
        save_viewer_yaml(cfg, p)
        cfg2 = load_viewer_yaml(p)
        assert cfg2.project.name == "roundtrip_test"
        assert cfg2.viewer.axis_order == cfg.viewer.axis_order


class TestLLMPatch:
    def test_patch_updates_label_layers(self, tmp_path):
        p = _write_yaml(tmp_path, {})
        cfg = load_viewer_yaml(p)
        original_count = len(cfg.label_layers)

        patch = {
            "label_layers": [
                {"name": "new_layer", "color": "#00ff00", "classes": ["background", "thing"]}
            ]
        }
        cfg2 = apply_patch(cfg, patch)
        # Patch replaces the list (deep-merge for lists replaces them)
        assert any(l.name == "new_layer" for l in cfg2.label_layers)

    def test_patch_updates_nested_key(self, tmp_path):
        p = _write_yaml(tmp_path, {})
        cfg = load_viewer_yaml(p)
        cfg2 = apply_patch(cfg, {"viewer": {"axis_order": "ZYX"}})
        assert cfg2.viewer.axis_order == "ZYX"
