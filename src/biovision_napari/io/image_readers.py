"""
Lazy image readers for 8 biological image formats.
All readers return a dask array with axes transposed to the project axis order.

Supported formats:
  zarr        — zarr.open()
  tif/tiff    — tifffile + dask
  n5          — zarr with N5Store
  h5/hdf5     — h5py + dask
  czi         — aicsimageio (AICSImage)
  lif         — aicsimageio (AICSImage)
  nd2         — nd2 library
  ome.tiff    — tifffile (OME metadata parsed for axis order)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import dask.array as da
import numpy as np


# Axis transposition helpers
# ---------------------------------------------------------------------------

_AXIS_CHARS = list("TCZYX")


def transpose_to_target(arr: da.Array, source_axes: str, target_axes: str) -> da.Array:
    """
    Reorder and/or insert size-1 axes so that arr goes from source_axes to target_axes.

    Example: source=CZYX, target=TZYX → insert T at front, result shape (1, C, Z, Y, X)
    then squeeze/reorder to (T=1, Z, Y, X) if C is not in target.

    This is best-effort: axes in source but not in target are squeezed (if size==1)
    or kept as the first non-target dims. Axes in target but not in source are
    inserted at size 1.
    """
    src = source_axes.upper()
    tgt = target_axes.upper()

    if src == tgt:
        return arr

    # Step 1: Insert missing target axes as size-1 dims
    current_axes = list(src)
    current_arr = arr
    for ax in tgt:
        if ax not in current_axes:
            current_axes.insert(0, ax)
            current_arr = da.expand_dims(current_arr, axis=0)

    # Step 2: Build transpose order for target
    try:
        perm = [current_axes.index(ax) for ax in tgt]
    except ValueError as exc:
        raise ValueError(
            f"Cannot transpose axes {src!r} to {tgt!r}: {exc}"
        ) from exc

    return current_arr.transpose(perm)


# ---------------------------------------------------------------------------
# Individual readers
# ---------------------------------------------------------------------------

def read_zarr(path: Path, axis_order: str) -> tuple[da.Array, str]:
    """Read a zarr array. Returns (dask_array, detected_axes)."""
    import zarr

    store = zarr.open(str(path), mode="r")
    # zarr root may be a group; pick first array
    if isinstance(store, zarr.Group):
        arrays = [v for v in store.values() if isinstance(v, zarr.Array)]
        if not arrays:
            raise ValueError(f"No arrays found in zarr store: {path}")
        z = arrays[0]
    else:
        z = store

    arr = da.from_zarr(z)
    detected = _guess_axes_from_ndim(arr.ndim)
    return arr, detected


def read_n5(path: Path, axis_order: str) -> tuple[da.Array, str]:
    """Read an N5 store (zarr with N5Store backend)."""
    import zarr
    from zarr.n5 import N5Store

    store = zarr.open(N5Store(str(path)), mode="r")
    if isinstance(store, zarr.Group):
        arrays = [v for v in store.values() if isinstance(v, zarr.Array)]
        if not arrays:
            raise ValueError(f"No arrays in N5 store: {path}")
        z = arrays[0]
    else:
        z = store

    arr = da.from_zarr(z)
    detected = _guess_axes_from_ndim(arr.ndim)
    return arr, detected


def read_tiff(path: Path, axis_order: str) -> tuple[da.Array, str]:
    """Read a TIFF (regular or OME-TIFF) lazily with tifffile."""
    import tifffile

    store = tifffile.imread(str(path), aszarr=True)
    arr = da.from_zarr(store)

    # Try to read OME-XML axis metadata
    detected = _read_ome_axes(path) or _guess_axes_from_ndim(arr.ndim)
    return arr, detected


def read_ome_tiff(path: Path, axis_order: str) -> tuple[da.Array, str]:
    """Read OME-TIFF; same as tiff but enforces OME metadata parsing."""
    return read_tiff(path, axis_order)


def read_hdf5(path: Path, axis_order: str, dataset_key: str = "data") -> tuple[da.Array, str]:
    """
    Read an HDF5 file lazily. Tries common dataset keys:
    'data', 'raw', 'volume', first dataset found.
    """
    import h5py

    def _find_first_dataset(f: h5py.File) -> Optional[str]:
        found = []

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                found.append(name)

        f.visititems(visitor)
        return found[0] if found else None

    with h5py.File(str(path), "r") as f:
        key = None
        for candidate in [dataset_key, "data", "raw", "volume", "image"]:
            if candidate in f:
                key = candidate
                break
        if key is None:
            key = _find_first_dataset(f)
        if key is None:
            raise ValueError(f"No dataset found in HDF5 file: {path}")

        dataset = f[key]
        arr = da.from_array(dataset, chunks="auto")
        arr = arr.persist()  # keep lazy but materialise the graph node

    detected = _guess_axes_from_ndim(arr.ndim)
    return arr, detected


def read_czi(path: Path, axis_order: str) -> tuple[da.Array, str]:
    """Read Zeiss CZI via aicsimageio."""
    from aicsimageio import AICSImage

    img = AICSImage(str(path))
    arr = img.get_dask_data()        # returns TCZYX by convention
    detected = "TCZYX"
    return arr, detected


def read_lif(path: Path, axis_order: str) -> tuple[da.Array, str]:
    """Read Leica LIF via aicsimageio."""
    from aicsimageio import AICSImage

    img = AICSImage(str(path))
    arr = img.get_dask_data()
    detected = "TCZYX"
    return arr, detected


def read_nd2(path: Path, axis_order: str) -> tuple[da.Array, str]:
    """Read Nikon ND2 via the nd2 library."""
    import nd2

    f = nd2.ND2File(str(path))
    arr = f.to_dask()
    # nd2 axis order varies; use axes attribute if available
    axes_str = getattr(f, "sizes", {})
    if axes_str:
        detected = "".join(k.upper() for k in axes_str.keys())
    else:
        detected = _guess_axes_from_ndim(arr.ndim)
    return arr, detected


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SUFFIX_MAP = {
    ".zarr": read_zarr,
    ".n5": read_n5,
    ".tif": read_tiff,
    ".tiff": read_tiff,
    ".h5": read_hdf5,
    ".hdf5": read_hdf5,
    ".czi": read_czi,
    ".lif": read_lif,
    ".nd2": read_nd2,
}


def load_image(path: str | Path, target_axis_order: str = "TZYX") -> da.Array:
    """
    Load any supported biological image format and return a dask array
    transposed to target_axis_order.
    """
    path = Path(path)

    # OME-TIFF detection: name ends with .ome.tif or .ome.tiff
    name_lower = path.name.lower()
    if name_lower.endswith(".ome.tif") or name_lower.endswith(".ome.tiff"):
        reader = read_ome_tiff
    else:
        suffix = path.suffix.lower()
        reader = _SUFFIX_MAP.get(suffix)
        if reader is None:
            raise ValueError(
                f"Unsupported format: {suffix!r}. "
                f"Supported: {list(_SUFFIX_MAP.keys())}"
            )

    arr, detected_axes = reader(path, target_axis_order)
    arr = transpose_to_target(arr, detected_axes, target_axis_order)
    return arr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guess_axes_from_ndim(ndim: int) -> str:
    """Best-effort axis label assignment based on number of dimensions."""
    defaults = {
        2: "YX",
        3: "ZYX",
        4: "TZYX",
        5: "TCZYX",
    }
    return defaults.get(ndim, "TZYX"[:ndim] if ndim <= 4 else "X" * ndim)


def _read_ome_axes(path: Path) -> Optional[str]:
    """Try to extract axes string from OME-XML embedded in TIFF."""
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as tif:
            if tif.is_ome and tif.ome_metadata:
                # Parse axes from OME-XML
                match = re.search(r'DimensionOrder="([A-Z]+)"', tif.ome_metadata)
                if match:
                    ome_order = match.group(1)  # e.g. "XYZCT"
                    # OME stores fastest-varying last; reverse for numpy order
                    return ome_order[::-1]
    except Exception:
        pass
    return None
