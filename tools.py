"""
Shared LangChain tool library for the BioVision SAM2 inference pipeline.

Tools
-----
load_sam2_model         — download + cache SAM2 models from HuggingFace
discover_samples        — scan a directory tree for image files
select_inference_mode   — choose 'auto' vs 'video' SAM2 mode from volume metadata
run_sam2_on_volume      — run SAM2 segmentation on a 4-D TZYX volume
save_biovision_masks    — write versioned masks (replicates mask_io logic inline)

The versioning helpers (_next_version_tag, _sha256) are replicated from
UI/src/biovision_napari/io/mask_io.py so this file is self-contained and
does not require the napari environment.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile
from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Module-level model cache
# {model_size: {"auto": SAM2AutomaticMaskGenerator,
#               "predictor": SAM2ImagePredictor,
#               "video": SAM2VideoPredictor,
#               "device": str}}
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, dict] = {}

# Image extensions recognised during sample discovery
_IMAGE_EXTS = {".tif", ".tiff", ".zarr", ".h5", ".hdf5", ".nd2", ".czi", ".lif"}


# ---------------------------------------------------------------------------
# Versioning helpers (replicated from mask_io.py)
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


def _sha256(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Internal image helpers
# ---------------------------------------------------------------------------

def _to_uint8_rgb(arr_2d: np.ndarray) -> np.ndarray:
    """Normalise a 2-D grayscale slice to uint8 and stack into 3-channel RGB."""
    a = arr_2d.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi > lo:
        a = (a - lo) / (hi - lo)
    a = (a * 255).astype(np.uint8)
    return np.stack([a, a, a], axis=-1)


def _merge_sam2_masks(masks_list: list[dict], h: int, w: int) -> np.ndarray:
    """Merge SAM2 auto-mode mask dicts (each has 'segmentation') into a label array."""
    label = np.zeros((h, w), dtype=np.int32)
    for idx, m in enumerate(masks_list, start=1):
        label[m["segmentation"]] = idx
    return label


def _ensure_4d(arr: np.ndarray) -> np.ndarray:
    """Normalise any array to exactly 4 dimensions (T, Z, Y, X)."""
    if arr.ndim == 2:
        return arr[np.newaxis, np.newaxis]          # T=1, Z=1
    if arr.ndim == 3:
        return arr[np.newaxis]                       # T=1
    while arr.ndim > 4:
        arr = arr[0]
    return arr


# ---------------------------------------------------------------------------
# Tool 1: load_sam2_model
# ---------------------------------------------------------------------------

@tool
def load_sam2_model(model_size: str = "tiny") -> str:
    """Load SAM2 models from HuggingFace and cache them for inference.

    Downloads SAM2AutomaticMaskGenerator, SAM2ImagePredictor, and
    SAM2VideoPredictor for the given model size. Subsequent calls for the
    same size return immediately from cache.

    Args:
        model_size: One of 'tiny', 'small', 'base-plus', 'large'.

    Returns:
        JSON string: {"model_size": str, "device": str, "status": "loaded"|"cached"}
    """
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    if model_size in _MODEL_CACHE:
        return json.dumps({
            "model_size": model_size,
            "device": _MODEL_CACHE[model_size]["device"],
            "status": "cached",
        })

    hf_repo = f"facebook/sam2-hiera-{model_size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predictor = SAM2ImagePredictor.from_pretrained(hf_repo, device=device)
    auto_gen = SAM2AutomaticMaskGenerator(predictor.model)
    video_pred = SAM2VideoPredictor.from_pretrained(hf_repo, device=device)

    _MODEL_CACHE[model_size] = {
        "auto": auto_gen,
        "predictor": predictor,
        "video": video_pred,
        "device": device,
    }

    return json.dumps({"model_size": model_size, "device": device, "status": "loaded"})


# ---------------------------------------------------------------------------
# Tool 2: discover_samples
# ---------------------------------------------------------------------------

@tool
def discover_samples(source_dir: str) -> str:
    """Scan subdirectories of source_dir for image files.

    Each immediate subdirectory is treated as one sample. The first image
    file found (by sorted name) in each subdirectory is used.

    Args:
        source_dir: Root directory to scan.

    Returns:
        JSON list: [{"sample_id": str, "image_path": str}, ...]
    """
    root = Path(source_dir)
    if not root.exists():
        return json.dumps([])

    samples = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        for fpath in sorted(sub.iterdir()):
            if fpath.suffix.lower() in _IMAGE_EXTS:
                samples.append({
                    "sample_id": sub.name,
                    "image_path": str(fpath),
                })
                break  # first matching file only

    return json.dumps(samples)


# ---------------------------------------------------------------------------
# Tool 3: select_inference_mode
# ---------------------------------------------------------------------------

@tool
def select_inference_mode(image_path: str, has_prompts: bool = False) -> str:
    """Determine whether to use SAM2 'auto' or 'video' mode for a volume.

    Reads the image at image_path, normalises it to 4-D (T, Z, Y, X), then
    applies the rule: if Z > 1 AND has_prompts → 'video', else → 'auto'.

    Args:
        image_path: Path to the image file (tif supported; other formats may work).
        has_prompts: True if the caller has point/box prompts to supply.

    Returns:
        JSON string: {"mode": str, "shape": list, "n_z_slices": int, "t_frames": int}
    """
    arr = _ensure_4d(tifffile.imread(image_path))
    t_frames, n_z_slices = arr.shape[0], arr.shape[1]
    mode = "video" if (n_z_slices > 1 and has_prompts) else "auto"

    return json.dumps({
        "mode": mode,
        "shape": list(arr.shape),
        "n_z_slices": n_z_slices,
        "t_frames": t_frames,
    })


# ---------------------------------------------------------------------------
# Tool 4: run_sam2_on_volume
# ---------------------------------------------------------------------------

@tool
def run_sam2_on_volume(
    image_path: str,
    output_dir: str,
    sample_id: str,
    layer_name: str,
    model_size: str = "tiny",
    mode: str = "auto",
    prompt_z: int = 0,
    point_prompts_json: str = "null",
    box_prompt_json: str = "null",
) -> str:
    """Run SAM2 segmentation on a 4-D (T, Z, Y, X) bioimage volume.

    auto mode:
        For each T frame and each Z slice, convert to uint8 RGB and run
        SAM2AutomaticMaskGenerator. Instance masks are merged into a label array.

    video mode:
        For each T frame, write Z slices as JPEG frames in a temp directory,
        initialise SAM2VideoPredictor, add point/box prompts at prompt_z,
        then propagate masks across all Z slices.

    The model is loaded from _MODEL_CACHE (call load_sam2_model first).

    Args:
        image_path:          Path to the input image volume.
        output_dir:          Directory where the raw mask TIFF is written.
        sample_id:           Sample identifier string.
        layer_name:          Segmentation layer name (e.g. 'cells').
        model_size:          SAM2 model size key matching _MODEL_CACHE.
        mode:                'auto' or 'video'.
        prompt_z:            Z index where prompts are defined (video mode only).
        point_prompts_json:  JSON list [{"coords": [x, y], "label": 1}, ...] or "null".
        box_prompt_json:     JSON list [x1, y1, x2, y2] or "null".

    Returns:
        JSON string: {"mask_path": str, "shape": list, "mode": str, "n_objects": int}
    """
    import torch
    from PIL import Image as PILImage

    # Ensure model is loaded
    if model_size not in _MODEL_CACHE:
        load_sam2_model.invoke({"model_size": model_size})
    cache = _MODEL_CACHE[model_size]

    arr = _ensure_4d(tifffile.imread(image_path))
    T, Z, H, W = arr.shape
    mask_volume = np.zeros((T, Z, H, W), dtype=np.int32)

    if mode == "auto":
        auto_gen = cache["auto"]
        for t in range(T):
            for z in range(Z):
                rgb = _to_uint8_rgb(arr[t, z])
                with torch.inference_mode():
                    masks = auto_gen.generate(rgb)
                mask_volume[t, z] = _merge_sam2_masks(masks, H, W)

    else:  # video mode
        video_pred = cache["video"]
        point_prompts = json.loads(point_prompts_json) if point_prompts_json != "null" else None
        box_prompt = json.loads(box_prompt_json) if box_prompt_json != "null" else None

        for t in range(T):
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write Z slices as JPEG frames (SAM2VideoPredictor expects image files)
                for z in range(Z):
                    rgb = _to_uint8_rgb(arr[t, z])
                    PILImage.fromarray(rgb).save(
                        os.path.join(tmpdir, f"{z:05d}.jpg"),
                        quality=95,
                    )

                with torch.inference_mode():
                    state = video_pred.init_state(video_path=tmpdir)
                    video_pred.reset_state(state)
                    ann_obj_id = 1

                    if point_prompts:
                        pts = np.array(
                            [[p["coords"][0], p["coords"][1]] for p in point_prompts],
                            dtype=np.float32,
                        )
                        lbls = np.array(
                            [p["label"] for p in point_prompts],
                            dtype=np.int32,
                        )
                        video_pred.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=prompt_z,
                            obj_id=ann_obj_id,
                            points=pts,
                            labels=lbls,
                        )
                    elif box_prompt:
                        video_pred.add_new_points_or_box(
                            inference_state=state,
                            frame_idx=prompt_z,
                            obj_id=ann_obj_id,
                            box=np.array(box_prompt, dtype=np.float32),
                        )

                    for frame_idx, obj_ids, mask_logits in video_pred.propagate_in_video(state):
                        for oi, logits in zip(obj_ids, mask_logits):
                            seg = (logits[0] > 0.0).cpu().numpy()
                            mask_volume[t, frame_idx][seg] = int(oi)

    out_path = Path(output_dir) / f"{sample_id}__{layer_name}.tif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), mask_volume.astype(np.uint32))

    return json.dumps({
        "mask_path": str(out_path),
        "shape": list(mask_volume.shape),
        "mode": mode,
        "n_objects": int(mask_volume.max()),
    })


# ---------------------------------------------------------------------------
# Tool 5: save_biovision_masks
# ---------------------------------------------------------------------------

@tool
def save_biovision_masks(
    sample_id: str,
    temp_layer_paths_json: str,
    masks_root: str,
) -> str:
    """Save versioned BioVision masks for one sample.

    Reads each temp TIFF back into numpy, then writes versioned copies to:
      masks_root/versions/v{N:04d}/{sample_id}__{layer}.tif
    along with a manifest.json recording sha256, shape, and dtype.

    This replicates mask_io.save_masks() inline so tools.py is self-contained
    and does not require the napari/biovision_napari package to be installed.

    Args:
        sample_id:             Sample identifier string.
        temp_layer_paths_json: JSON dict {"layer_name": "/path/to/tmp.tif", ...}.
        masks_root:            Root of the masks store (paths.masks in viewer.yaml).

    Returns:
        JSON string: {"version": str, "masks_dir": str, "layers_saved": list}
    """
    layer_paths: dict[str, str] = json.loads(temp_layer_paths_json)

    versions_root = Path(masks_root) / "versions"
    versions_root.mkdir(parents=True, exist_ok=True)

    version_tag = _next_version_tag(versions_root)
    version_dir = versions_root / version_tag
    version_dir.mkdir()

    manifest_layers = []
    for layer_name, tmp_path in layer_paths.items():
        arr = tifffile.imread(tmp_path)
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

    return json.dumps({
        "version": version_tag,
        "masks_dir": str(version_dir),
        "layers_saved": list(layer_paths.keys()),
    })
