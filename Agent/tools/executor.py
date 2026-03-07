"""
Sandboxed Python execution for generated preprocessing scripts.

Security model
--------------
Subprocess isolation (default — dev / single-tenant)
    The generated script runs in a child process with a minimal, explicit
    environment.  Only INPUT_DIR, OUTPUT_DIR, and PATH are forwarded; no API
    keys, shell history, or other secrets can leak through.  A hard timeout
    prevents runaway processes.

Docker isolation (recommended for production / multi-tenant deployments)
    See `_exec_docker` below.  Switch by passing `use_docker=True` to
    `exec_sandboxed()` or by setting the env-var BIOVISION_USE_DOCKER=1.
    You must build a locked-down container image first:

        FROM python:3.11-slim
        RUN pip install numpy scikit-image tifffile Pillow scipy
        # No network access, read-only root FS except /data/output

    Then mount input as read-only and output as writable:
        docker run --rm --network none \\
            -v <input_dir>:/data/input:ro \\
            -v <output_dir>:/data/output \\
            -v <script>:/script.py:ro \\
            -e INPUT_DIR=/data/input -e OUTPUT_DIR=/data/output \\
            biovision-sandbox:latest python /script.py
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("biovision.executor")

# ── Dependency installation ───────────────────────────────────────────────────


def _ensure_dependencies(packages: list[str]) -> Optional[str]:
    """
    pip-install *packages* unconditionally.

    Returns
    -------
    None
        All packages installed successfully.
    str
        Error message if pip install fails — caller should surface this to the
        Coder LLM so it can correct any hallucinated package names.
    """
    if not packages:
        return None

    logger.info("Executor: installing dependencies: %s", packages)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", *packages],
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        err = f"pip install failed for {packages} (exit code {exc.returncode})"
        logger.error("Executor: %s", err)
        return err
    except subprocess.TimeoutExpired:
        err = f"pip install timed out after 120 s for {packages}"
        logger.error("Executor: %s", err)
        return err
    except Exception as exc:  # noqa: BLE001
        err = f"pip install error: {exc}"
        logger.error("Executor: %s", err)
        return err

    return None

# ── Execution back-ends ───────────────────────────────────────────────────────


def _exec_subprocess(
    script_path: str,
    input_dir: str,
    output_dir: str,
    timeout: int,
) -> dict:
    """Run *script_path* in a child process with a minimal environment."""
    env = {
        "INPUT_DIR": input_dir,
        "OUTPUT_DIR": output_dir,
        # Forward PATH so the child can locate system binaries (e.g. libGL).
        "PATH": os.environ.get("PATH", ""),
        # Keep PYTHONPATH so the subprocess can find installed site-packages.
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return {
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "success": result.returncode == 0,
    }


def _exec_docker(
    script_path: str,
    input_dir: str,
    output_dir: str,
    timeout: int,
) -> dict:  # pragma: no cover
    """
    Run the script inside a locked-down Docker container.

    Prerequisites
    -------------
    1. Build the sandbox image:   docker build -t biovision-sandbox .
    2. Ensure the Docker daemon is running and the current user has access.

    TODO: replace the image name and adjust volume paths for your deployment.
    """
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",            # no outbound network
        "--memory", "2g",               # cap RAM
        "--cpus", "2",
        "-v", f"{input_dir}:/data/input:ro",
        "-v", f"{output_dir}:/data/output",
        "-v", f"{script_path}:/script.py:ro",
        "-e", "INPUT_DIR=/data/input",
        "-e", "OUTPUT_DIR=/data/output",
        "biovision-sandbox:latest",
        "python", "/script.py",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "success": result.returncode == 0,
    }


# ── Public API ────────────────────────────────────────────────────────────────


def exec_sandboxed(
    code: str,
    input_dir: str,
    output_dir: str,
    dependencies: Optional[list[str]] = None,
    timeout: int = 300,
    use_docker: bool = False,
) -> dict:
    """
    Write *code* to a temp file and execute it in an isolated subprocess.

    Parameters
    ----------
    code:
        Raw Python source to execute.
    input_dir:
        Directory containing source images; forwarded as INPUT_DIR env-var.
    output_dir:
        Directory where the script must write results; forwarded as OUTPUT_DIR.
    dependencies:
        pip package names to install before running (if not already present).
    timeout:
        Hard wall-clock limit in seconds (default 300 s / 5 min).
    use_docker:
        Set True to use the Docker back-end instead of subprocess.

    Returns
    -------
    dict
        ``{"success": bool, "stdout": str, "stderr": str}``
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    dep_error = _ensure_dependencies(dependencies or [])
    if dep_error:
        return {"success": False, "stdout": "", "stderr": dep_error}

    with tempfile.NamedTemporaryFile(
        suffix=".py", delete=False, mode="w", encoding="utf-8"
    ) as fh:
        fh.write(code)
        script_path = fh.name

    try:
        backend = _exec_docker if use_docker else _exec_subprocess
        return backend(script_path, input_dir, output_dir, timeout)

    except subprocess.TimeoutExpired:
        logger.error("Executor: script timed out after %d s.", timeout)
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout} seconds.",
        }
    except FileNotFoundError as exc:
        # Docker binary not found, etc.
        logger.exception("Executor: backend not found.")
        return {"success": False, "stdout": "", "stderr": str(exc)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Executor: unexpected error.")
        return {"success": False, "stdout": "", "stderr": str(exc)}
    finally:
        os.unlink(script_path)
