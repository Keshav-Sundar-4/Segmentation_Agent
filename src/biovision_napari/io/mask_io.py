"""
Versioned mask I/O.

Masks are saved to:
  <masks_root>/versions/v{N:04d}/{sample_id}__{layer}.tif

Each version folder also contains a manifest.json with:
  version, created_at, sample_id, layers (name, file, sha256, shape, dtype)
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile


# ---------------------------------------------------------------------------
# Versioning helpers
# ---------------------------------------------------------------------------

def _next_version_tag(versions_root: Path) -> str:
    """Return the next version tag, e.g. 'v0001' if no versions exist yet."""
    if not versions_root.exists():
        return "v0001"
    existing = [
        d.name for d in versions_root.iterdir()
        if d.is_dir() and re.match(r"v\d{4,}", d.name)
    ]
    if not existing:
        return "v0001"
    nums = [int(re.search(r"\d+", v).group()) for v in existing]
    return f"v{max(nums) + 1:04d}"


def _latest_version_tag(versions_root: Path) -> Optional[str]:
    """Return the tag of the most recent version folder, or None."""
    if not versions_root.exists():
        return None
    existing = [
        d.name for d in versions_root.iterdir()
        if d.is_dir() and re.match(r"v\d{4,}", d.name)
    ]
    if not existing:
        return None
    return sorted(existing)[-1]


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_masks(
    sample_id: str,
    masks_root: str | Path,
    layers: dict[str, np.ndarray],
) -> str:
    """
    Save one mask per label layer in a new version folder.

    Parameters
    ----------
    sample_id : str
    masks_root : Path
        Root of GT masks (paths.masks in viewer.yaml).
    layers : dict
        Mapping of layer_name → numpy label array.

    Returns
    -------
    str
        The version tag that was created (e.g. 'v0003').
    """
    masks_root = Path(masks_root)
    versions_root = masks_root / "versions"
    versions_root.mkdir(parents=True, exist_ok=True)

    version_tag = _next_version_tag(versions_root)
    version_dir = versions_root / version_tag
    version_dir.mkdir()

    manifest_layers = []
    for layer_name, arr in layers.items():
        filename = f"{sample_id}__{layer_name}.tif"
        fpath = version_dir / filename
        tifffile.imwrite(str(fpath), arr.astype(np.uint32))
        manifest_layers.append({
            "layer": layer_name,
            "file": filename,
            "sha256": _sha256(arr),
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        })

    manifest = {
        "version": version_tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_id": sample_id,
        "layers": manifest_layers,
    }
    (version_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return version_tag


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_masks(
    sample_id: str,
    masks_root: str | Path,
    version_tag: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """
    Load masks for a sample from a specific version (or the latest).

    Returns
    -------
    dict
        Mapping of layer_name → numpy array. Empty dict if no masks found.
    """
    masks_root = Path(masks_root)
    versions_root = masks_root / "versions"

    if version_tag:
        tag = version_tag
    else:
        # Find latest version that actually contains this sample
        sample_versions = list_versions(sample_id, masks_root)
        tag = sample_versions[-1] if sample_versions else None

    if tag is None:
        return {}

    version_dir = versions_root / tag
    if not version_dir.exists():
        return {}

    # Read manifest to get layer list (authoritative)
    manifest_path = version_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = {}
        for layer_info in manifest.get("layers", []):
            fpath = version_dir / layer_info["file"]
            if fpath.exists():
                result[layer_info["layer"]] = tifffile.imread(str(fpath))
        return result

    # Fallback: discover by naming convention if manifest missing
    result = {}
    prefix = f"{sample_id}__"
    for tif_file in version_dir.glob(f"{prefix}*.tif"):
        layer_name = tif_file.stem[len(prefix):]
        result[layer_name] = tifffile.imread(str(tif_file))
    return result


def list_versions(sample_id: str, masks_root: str | Path) -> list[str]:
    """
    Return sorted list of version tags that contain masks for this sample.
    """
    masks_root = Path(masks_root)
    versions_root = masks_root / "versions"
    if not versions_root.exists():
        return []

    tags = []
    for vdir in sorted(versions_root.iterdir()):
        if not (vdir.is_dir() and re.match(r"v\d{4,}", vdir.name)):
            continue
        prefix = f"{sample_id}__"
        if any(f.name.startswith(prefix) for f in vdir.glob("*.tif")):
            tags.append(vdir.name)
    return tags
