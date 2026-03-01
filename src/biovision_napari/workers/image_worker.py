"""
Background worker for loading images without blocking the UI thread.
Uses napari's thread_worker pattern.
"""
from __future__ import annotations

from pathlib import Path

from napari.qt.threading import thread_worker


@thread_worker
def load_image_worker(path: str | Path, axis_order: str):
    """
    Load an image in a background thread and yield the dask array.
    Connect worker.yielded to receive the result.
    """
    from biovision_napari.io.image_readers import load_image

    arr = load_image(path, target_axis_order=axis_order)
    yield arr
