"""Tests for mask save/load roundtrip correctness."""
import numpy as np
import pytest

from biovision_napari.io.mask_io import save_masks, load_masks


class TestMaskRoundtrip:
    def _make_layers(self) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(42)
        return {
            "cells": rng.integers(0, 5, size=(8, 16, 16), dtype=np.uint32),
            "nuclei": rng.integers(0, 3, size=(8, 16, 16), dtype=np.uint32),
        }

    def test_roundtrip_values_preserved(self, tmp_path):
        masks_root = tmp_path / "masks"
        original = self._make_layers()
        save_masks("sample_a", masks_root, original)
        loaded = load_masks("sample_a", masks_root)

        assert set(loaded.keys()) == set(original.keys())
        for name in original:
            np.testing.assert_array_equal(
                loaded[name], original[name].astype(np.uint32),
                err_msg=f"Layer {name!r} did not roundtrip correctly"
            )

    def test_roundtrip_dtype(self, tmp_path):
        masks_root = tmp_path / "masks"
        layers = {"cells": np.array([[1, 2], [3, 0]], dtype=np.uint32)}
        save_masks("s", masks_root, layers)
        loaded = load_masks("s", masks_root)
        assert loaded["cells"].dtype == np.uint32

    def test_load_specific_version(self, tmp_path):
        masks_root = tmp_path / "masks"
        v1_layers = {"cells": np.zeros((4, 4), dtype=np.uint32)}
        v2_layers = {"cells": np.ones((4, 4), dtype=np.uint32) * 99}

        save_masks("s", masks_root, v1_layers)
        save_masks("s", masks_root, v2_layers)

        loaded_v1 = load_masks("s", masks_root, version_tag="v0001")
        loaded_v2 = load_masks("s", masks_root, version_tag="v0002")

        np.testing.assert_array_equal(loaded_v1["cells"], v1_layers["cells"])
        np.testing.assert_array_equal(loaded_v2["cells"], v2_layers["cells"])

    def test_load_latest_by_default(self, tmp_path):
        masks_root = tmp_path / "masks"
        save_masks("s", masks_root, {"cells": np.zeros((4, 4), dtype=np.uint32)})
        save_masks("s", masks_root, {"cells": np.full((4, 4), 7, dtype=np.uint32)})

        loaded = load_masks("s", masks_root)
        np.testing.assert_array_equal(loaded["cells"], np.full((4, 4), 7, dtype=np.uint32))

    def test_load_missing_sample_returns_empty(self, tmp_path):
        masks_root = tmp_path / "masks"
        result = load_masks("nonexistent_sample", masks_root)
        assert result == {}

    def test_multiple_samples_isolated(self, tmp_path):
        masks_root = tmp_path / "masks"
        a = {"cells": np.full((4, 4), 1, dtype=np.uint32)}
        b = {"cells": np.full((4, 4), 2, dtype=np.uint32)}
        save_masks("sample_a", masks_root, a)
        save_masks("sample_b", masks_root, b)

        loaded_a = load_masks("sample_a", masks_root)
        loaded_b = load_masks("sample_b", masks_root)

        np.testing.assert_array_equal(loaded_a["cells"], a["cells"])
        np.testing.assert_array_equal(loaded_b["cells"], b["cells"])
