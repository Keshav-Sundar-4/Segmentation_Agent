"""Tests for versioned mask folder increment logic."""
import json
from pathlib import Path

import numpy as np
import pytest

from biovision_napari.io.mask_io import (
    _next_version_tag,
    _latest_version_tag,
    save_masks,
    list_versions,
)


class TestVersionTag:
    def test_first_version(self, tmp_path):
        versions_root = tmp_path / "versions"
        assert _next_version_tag(versions_root) == "v0001"

    def test_increments_from_existing(self, tmp_path):
        versions_root = tmp_path / "versions"
        versions_root.mkdir()
        (versions_root / "v0001").mkdir()
        (versions_root / "v0002").mkdir()
        assert _next_version_tag(versions_root) == "v0003"

    def test_handles_gaps(self, tmp_path):
        versions_root = tmp_path / "versions"
        versions_root.mkdir()
        (versions_root / "v0001").mkdir()
        (versions_root / "v0005").mkdir()
        assert _next_version_tag(versions_root) == "v0006"

    def test_latest_returns_last(self, tmp_path):
        versions_root = tmp_path / "versions"
        versions_root.mkdir()
        (versions_root / "v0001").mkdir()
        (versions_root / "v0003").mkdir()
        assert _latest_version_tag(versions_root) == "v0003"

    def test_latest_returns_none_when_empty(self, tmp_path):
        versions_root = tmp_path / "versions"
        assert _latest_version_tag(versions_root) is None


class TestSaveMasks:
    def _dummy_layers(self):
        return {
            "cells": np.zeros((10, 10), dtype=np.uint32),
            "nuclei": np.ones((10, 10), dtype=np.uint32),
        }

    def test_creates_version_folder(self, tmp_path):
        masks_root = tmp_path / "masks"
        tag = save_masks("sample_01", masks_root, self._dummy_layers())
        assert tag == "v0001"
        assert (masks_root / "versions" / "v0001").is_dir()

    def test_saves_tif_per_layer(self, tmp_path):
        masks_root = tmp_path / "masks"
        save_masks("sample_01", masks_root, self._dummy_layers())
        v_dir = masks_root / "versions" / "v0001"
        assert (v_dir / "sample_01__cells.tif").exists()
        assert (v_dir / "sample_01__nuclei.tif").exists()

    def test_manifest_written(self, tmp_path):
        masks_root = tmp_path / "masks"
        save_masks("sample_01", masks_root, self._dummy_layers())
        manifest_path = masks_root / "versions" / "v0001" / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["version"] == "v0001"
        assert manifest["sample_id"] == "sample_01"
        assert len(manifest["layers"]) == 2

    def test_increments_on_second_save(self, tmp_path):
        masks_root = tmp_path / "masks"
        save_masks("sample_01", masks_root, self._dummy_layers())
        tag2 = save_masks("sample_01", masks_root, self._dummy_layers())
        assert tag2 == "v0002"

    def test_list_versions(self, tmp_path):
        masks_root = tmp_path / "masks"
        save_masks("s1", masks_root, self._dummy_layers())
        save_masks("s1", masks_root, self._dummy_layers())
        save_masks("s2", masks_root, {"cells": np.zeros((5, 5), dtype=np.uint32)})
        s1_versions = list_versions("s1", masks_root)
        assert s1_versions == ["v0001", "v0002"]
        s2_versions = list_versions("s2", masks_root)
        assert s2_versions == ["v0003"]
