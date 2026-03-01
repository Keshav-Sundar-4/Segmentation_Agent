"""
Sample discovery: scans dataset_root for subdirectories.
Each subdirectory is treated as one sample.
Status is persisted in a JSON sidecar inside the sample directory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


VALID_STATUSES = {"unlabeled", "in_progress", "done", "reviewed"}
DEFAULT_STATUS = "unlabeled"

# Image extensions we recognise when sniffing sample metadata
IMAGE_EXTENSIONS = {
    ".zarr", ".tif", ".tiff", ".n5", ".h5", ".hdf5",
    ".czi", ".lif", ".nd2",
}


@dataclass
class SampleInfo:
    sample_id: str
    path: Path
    status: str = DEFAULT_STATUS
    modality: str = "unknown"
    dims: str = ""
    has_gt_mask: bool = False
    models_available: list[str] = field(default_factory=list)

    def to_display_row(self) -> list[str]:
        return [
            self.sample_id,
            self.status,
            self.modality,
            self.dims,
            "yes" if self.has_gt_mask else "no",
            ", ".join(self.models_available) if self.models_available else "—",
        ]


def discover_samples(
    dataset_root: str | Path,
    status_file: str,
    masks_root: Optional[str | Path] = None,
) -> list[SampleInfo]:
    """
    Scan dataset_root for subdirectories. Each subdir = one sample.
    Reads status from <sample_dir>/<status_file> if present.
    Checks masks_root/versions/ to determine has_gt_mask.
    """
    root = Path(dataset_root)
    if not root.exists():
        return []

    masks = Path(masks_root) if masks_root else None
    samples: list[SampleInfo] = []

    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue

        sample_id = subdir.name
        info = SampleInfo(sample_id=sample_id, path=subdir)

        # Load status sidecar
        status_path = subdir / status_file
        if status_path.exists():
            try:
                sidecar = json.loads(status_path.read_text(encoding="utf-8"))
                info.status = sidecar.get("status", DEFAULT_STATUS)
                info.modality = sidecar.get("modality", "unknown")
                info.dims = sidecar.get("dims", "")
            except (json.JSONDecodeError, OSError):
                pass

        # Sniff modality / dims from image files if not set
        if info.modality == "unknown" or not info.dims:
            _sniff_image_metadata(info)

        # Check for GT masks
        if masks is not None:
            versions_dir = masks / "versions"
            if versions_dir.exists():
                # Look for any version folder containing this sample_id
                for vdir in versions_dir.iterdir():
                    if vdir.is_dir() and any(
                        f.stem.startswith(sample_id + "__") for f in vdir.glob("*.tif")
                    ):
                        info.has_gt_mask = True
                        break

        # Discover model outputs by convention: runs/<model>/<sample_id>/
        # (convention; populated once model pipeline is built)
        info.models_available = _discover_models(sample_id, root.parent)

        samples.append(info)

    return samples


def _sniff_image_metadata(info: SampleInfo) -> None:
    """Try to infer modality/dims from file extensions in the sample dir."""
    found_ext = None
    for f in info.path.rglob("*"):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            found_ext = f.suffix.lower()
            break
    if found_ext:
        ext_modality_map = {
            ".czi": "fluorescence",
            ".lif": "fluorescence",
            ".nd2": "fluorescence",
            ".tif": "unknown",
            ".tiff": "unknown",
            ".zarr": "unknown",
            ".n5": "unknown",
            ".h5": "unknown",
            ".hdf5": "unknown",
        }
        if info.modality == "unknown":
            info.modality = ext_modality_map.get(found_ext, "unknown")


def _discover_models(sample_id: str, project_root: Path) -> list[str]:
    """Find model output dirs by convention: <project_root>/runs/<model>/<sample_id>/."""
    runs = project_root / "runs"
    models = []
    if runs.exists():
        for model_dir in runs.iterdir():
            if model_dir.is_dir() and (model_dir / sample_id).exists():
                models.append(model_dir.name)
    return models


def write_sample_status(
    sample_path: Path,
    status_file: str,
    status: str,
    modality: str = "unknown",
    dims: str = "",
) -> None:
    """Persist sample status to the status sidecar JSON."""
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status {status!r}. Must be one of {VALID_STATUSES}")
    sidecar = {"status": status, "modality": modality, "dims": dims}
    path = sample_path / status_file
    path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
