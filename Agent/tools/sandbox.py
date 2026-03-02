"""
tools/sandbox.py — Sandboxed code execution for BioVision pipeline.

Two-path execution:
  Fast path  — all detected imports are available in current env → exec_repl(code)
  Slow path  — at least one import is unknown → temp venv, pip install, subprocess run

Public surface
──────────────
  exec_sandboxed(code) → str   dispatch: fast or slow path
  detect_imports(code) → set   top-level module names from AST
  IMPORT_TO_PIP: dict          canonical import name → pip package name
"""

from __future__ import annotations

import ast
import importlib
import logging
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from tools.repl import exec_repl

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mapping: import name → pip install name
# ─────────────────────────────────────────────────────────────────────────────

IMPORT_TO_PIP: dict[str, str] = {
    "cv2":            "opencv-python",
    "skimage":        "scikit-image",
    "PIL":            "Pillow",
    "tifffile":       "tifffile",
    "imageio":        "imageio",
    "zarr":           "zarr",
    "h5py":           "h5py",
    "torch":          "torch",
    "torchvision":    "torchvision",
    "scipy":          "scipy",
    "numpy":          "numpy",
    "matplotlib":     "matplotlib",
    "pandas":         "pandas",
    "sklearn":        "scikit-learn",
    "SimpleITK":      "SimpleITK",
    "sitk":           "SimpleITK",
    "nd2":            "nd2",
    "czifile":        "czifile",
    "aicsimageio":    "aicsimageio",
    "monai":          "monai",
    "albumentations": "albumentations",
    "pims":           "pims",
    "tqdm":           "tqdm",
}

# ─────────────────────────────────────────────────────────────────────────────
# Standard-library module names (never pip-install these)
# ─────────────────────────────────────────────────────────────────────────────

if hasattr(sys, "stdlib_module_names"):
    STDLIB_MODULES: frozenset = frozenset(sys.stdlib_module_names)
else:
    # Fallback for Python < 3.10
    STDLIB_MODULES = frozenset({
        "os", "sys", "re", "math", "io", "json", "pathlib", "typing", "abc",
        "ast", "copy", "datetime", "enum", "functools", "gc", "glob", "hashlib",
        "inspect", "itertools", "logging", "operator", "platform", "queue",
        "random", "shutil", "signal", "socket", "struct", "subprocess",
        "tempfile", "threading", "time", "traceback", "unittest", "uuid",
        "warnings", "weakref", "contextlib", "collections", "dataclasses",
        "importlib", "string", "textwrap", "types", "venv", "builtins",
    })

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def detect_imports(code: str) -> set[str]:
    """
    AST-parse *code* and return the set of top-level module names.

    e.g. ``import numpy as np; from skimage import filters`` → ``{"numpy", "skimage"}``
    """
    modules: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return modules

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module.split(".")[0])
    return modules


def _is_available(module: str) -> bool:
    """Return True if *module* can be imported from the current environment."""
    try:
        importlib.import_module(module)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def _pip_name(module: str) -> str:
    """Map an import name to the pip package name (falls back to the module name)."""
    return IMPORT_TO_PIP.get(module, module)


# ─────────────────────────────────────────────────────────────────────────────
# Slow path: temp venv + subprocess
# ─────────────────────────────────────────────────────────────────────────────


def _exec_in_venv(code: str, packages: list[str]) -> str:
    """
    Create a temp venv (inheriting current packages via system_site_packages),
    install *packages*, write *code* to a temp file, run it as a subprocess,
    and return stdout + stderr.  The temporary directory is cleaned up on exit.
    """
    with tempfile.TemporaryDirectory(prefix="bv_sandbox_") as tmpdir:
        venv_dir = Path(tmpdir) / ".venv"
        venv.create(str(venv_dir), with_pip=True, system_site_packages=True)

        # Locate the venv python executable
        venv_python = venv_dir / "bin" / "python"
        if not venv_python.exists():
            venv_python = venv_dir / "Scripts" / "python.exe"

        # Install only missing packages
        if packages:
            logger.info("Sandbox: pip-installing %s", packages)
            pip_result = subprocess.run(
                [str(venv_python), "-m", "pip", "install", "--quiet"] + packages,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if pip_result.returncode != 0:
                logger.warning("pip install stderr:\n%s", pip_result.stderr[:500])

        # Write code to a temp file and run it
        script_path = Path(tmpdir) / "bv_script.py"
        script_path.write_text(code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [str(venv_python), str(script_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            return "SANDBOX ERROR: script timed out after 300 seconds."

        output = proc.stdout + proc.stderr
        if proc.returncode != 0 and not output:
            output = f"SANDBOX ERROR: process exited with code {proc.returncode}"
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def exec_sandboxed(code: str) -> str:
    """
    Execute *code*, choosing the fastest safe path:

    Fast path  — all detected imports are in the current env → exec_repl(code)
    Slow path  — at least one import is unavailable → temp venv + subprocess

    Returns the combined stdout + stderr as a string.
    """
    imports     = detect_imports(code)
    third_party = imports - STDLIB_MODULES
    missing     = [m for m in third_party if not _is_available(m)]

    if not missing:
        logger.debug("Sandbox fast path: all imports available.")
        return exec_repl(code)

    packages = [_pip_name(m) for m in missing]
    logger.info("Sandbox slow path: installing %s for missing imports %s", packages, missing)
    return _exec_in_venv(code, packages)
