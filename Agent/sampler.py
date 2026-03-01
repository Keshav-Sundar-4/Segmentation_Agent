"""
Image discovery and mini-batch sampling utilities.

Used during the HITL evaluation phase to select a small, representative
subset of images so the agent can demonstrate a technique cheaply before
the user decides whether to apply it to the full dataset.
"""

import random
from pathlib import Path
from typing import List

IMAGE_EXTENSIONS: frozenset = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
)


def discover_images(folder: Path) -> List[Path]:
    """Return a sorted list of all image files found recursively in *folder*."""
    return sorted(
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def sample_images(folder: Path, n: int = 5, seed: int = 42) -> List[Path]:
    """
    Return up to *n* randomly sampled image paths from *folder*.

    Uses a fixed *seed* so the same subset is evaluated across retries
    within a single session (ensuring apples-to-apples comparison when
    the user rejects a technique and a new one is tried).

    Raises:
        ValueError: if no images are found in *folder*.
    """
    images = discover_images(folder)
    if not images:
        raise ValueError(f"No images found in '{folder}'.")
    rng = random.Random(seed)
    return rng.sample(images, min(n, len(images)))
