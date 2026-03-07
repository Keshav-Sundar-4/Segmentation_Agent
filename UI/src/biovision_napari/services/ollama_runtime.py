"""
Ollama runtime preflight helper.

Responsibilities
----------------
1. Detect whether the ``ollama`` CLI is installed.
2. Start the Ollama HTTP server if it is not already running.
3. Wait until the API is reachable.
4. Pull a requested model if it is not available locally.

This module is *runtime/infrastructure* code that runs BEFORE the LangGraph
pipeline starts.  It does NOT add any nodes to the graph.  All functions are
safe to call from a background thread (they block; do not call from the Qt
main thread directly).

Platform support
----------------
- macOS  : ``ollama serve`` is spawned in the background.
- Linux  : same.
- Windows: ``ollama serve`` is spawned with CREATE_NO_WINDOW so no console
           window flashes.  Tested with the official Ollama Windows installer.

Typical call from a worker thread
----------------------------------
    from biovision_napari.services.ollama_runtime import prepare_ollama

    resolved_model = prepare_ollama(
        model_name="llama3.2",
        base_url="http://localhost:11434",
        progress=lambda msg: print(msg),   # or feed to UI log
    )
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
import urllib.request
from typing import Callable, Optional

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL    = "llama3.2"

# How long to wait (seconds) for the server to become ready after launching it.
_SERVER_START_TIMEOUT  = 40.0
_SERVER_POLL_INTERVAL  = 0.5


# ── Low-level checks ──────────────────────────────────────────────────────────


def is_ollama_installed() -> bool:
    """Return True if the ``ollama`` binary is on PATH."""
    return shutil.which("ollama") is not None


def is_ollama_running(base_url: str = DEFAULT_BASE_URL) -> bool:
    """
    Return True if the Ollama HTTP API is reachable at *base_url*.
    Uses a short timeout so callers are not blocked for long.
    """
    try:
        url = base_url.rstrip("/") + "/api/tags"
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


# ── Server management ─────────────────────────────────────────────────────────


def start_ollama_server() -> subprocess.Popen:
    """
    Launch ``ollama serve`` as a background process and return its Popen handle.

    The process inherits no terminal and writes nothing to a console window.
    The caller does not need to manage the lifecycle; the server will stay
    running until the parent process (napari) exits.

    Raises
    ------
    RuntimeError
        If ``ollama`` is not found on PATH.
    """
    if not is_ollama_installed():
        raise RuntimeError(
            "Ollama is not installed on this machine. "
            "Install Ollama from https://ollama.com to use Local mode."
        )

    kwargs: dict = {
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if platform.system() == "Windows":
        # Prevent a console window from flashing on Windows.
        kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW

    return subprocess.Popen(["ollama", "serve"], **kwargs)


def wait_for_ollama_ready(
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = _SERVER_START_TIMEOUT,
    poll_interval: float = _SERVER_POLL_INTERVAL,
) -> None:
    """
    Block until the Ollama HTTP API at *base_url* is reachable, or raise.

    Raises
    ------
    TimeoutError
        If the server is not reachable within *timeout* seconds.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_ollama_running(base_url):
            return
        time.sleep(poll_interval)

    raise TimeoutError(
        f"Ollama did not become ready within {timeout:.0f} seconds. "
        "Check that Ollama is installed correctly and that port 11434 is not "
        "blocked by a firewall or another process."
    )


# ── Model management ──────────────────────────────────────────────────────────


def list_ollama_models(base_url: str = DEFAULT_BASE_URL) -> list[str]:
    """Return a list of locally available Ollama model names."""
    url = base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=4) as resp:
        data = json.loads(resp.read().decode())
    return [m["name"] for m in data.get("models", [])]


def _model_is_local(available: list[str], model_name: str) -> bool:
    """
    Return True if *model_name* is in the *available* list.

    Matches either the exact name (``"llama3.2:latest"``) or the bare name
    without a tag (``"llama3.2"`` matches ``"llama3.2:latest"``).
    """
    base = model_name.split(":")[0]
    for m in available:
        if m == model_name or m == base or m.startswith(base + ":"):
            return True
    return False


def ensure_ollama_model(
    model_name: str,
    base_url: str = DEFAULT_BASE_URL,
    progress: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Ensure *model_name* is available locally; pull it if not.

    Parameters
    ----------
    model_name:
        Ollama model name, e.g. ``"llama3.2"`` or ``"mistral:7b"``.
    base_url:
        Ollama base URL (used to check the local model list).
    progress:
        Optional callable that receives human-readable status strings.
        Called before and after the pull; not called during (pull is blocking).

    Raises
    ------
    RuntimeError
        If ``ollama pull`` exits with a non-zero return code.
    """
    # Check whether the model is already present.
    try:
        available = list_ollama_models(base_url)
        if _model_is_local(available, model_name):
            if progress:
                progress(f"Model '{model_name}' is already available locally.")
            return
    except Exception:
        # If we cannot query the list, attempt the pull anyway.
        pass

    if progress:
        progress(
            f"Model '{model_name}' not found locally. "
            "Pulling from Ollama registry — this may take several minutes…"
        )

    result = subprocess.run(
        ["ollama", "pull", model_name],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "unknown error").strip()
        raise RuntimeError(
            f"Failed to pull model '{model_name}': {err}. "
            "Check the model name and your internet connection."
        )

    if progress:
        progress(f"Model '{model_name}' pulled successfully.")


# ── High-level preflight ──────────────────────────────────────────────────────


def prepare_ollama(
    model_name: str,
    base_url: str = DEFAULT_BASE_URL,
    progress: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Full Ollama preflight sequence (call this before starting the pipeline):

    1. Verify ``ollama`` is installed — raise ``RuntimeError`` if not.
    2. Start the Ollama server if it is not already running.
    3. Wait until the HTTP API is reachable.
    4. Pull *model_name* if it is not available locally.

    Parameters
    ----------
    model_name:
        Ollama model name.  An empty string falls back to ``DEFAULT_MODEL``.
    base_url:
        Ollama base URL.
    progress:
        Optional callable receiving human-readable status strings, suitable
        for feeding into a UI log widget or a terminal.

    Returns
    -------
    str
        The resolved model name (non-empty; defaults to ``DEFAULT_MODEL``).

    Raises
    ------
    RuntimeError
        If Ollama is not installed or model pull fails.
    TimeoutError
        If the server does not become ready within the timeout.
    """
    resolved_model = model_name.strip() or DEFAULT_MODEL

    # ── Step 1: installed? ────────────────────────────────────────────────────
    if not is_ollama_installed():
        raise RuntimeError(
            "Ollama is not installed on this machine. "
            "Install Ollama from https://ollama.com to use Local mode."
        )

    # ── Step 2 & 3: server running? ───────────────────────────────────────────
    if is_ollama_running(base_url):
        if progress:
            progress("Ollama server is already running.")
    else:
        if progress:
            progress("Starting Ollama server…")
        start_ollama_server()
        # Brief pause so the process has time to bind the port.
        time.sleep(0.5)
        if progress:
            progress("Waiting for Ollama server to be ready…")
        wait_for_ollama_ready(base_url=base_url, timeout=_SERVER_START_TIMEOUT)
        if progress:
            progress("Ollama server started successfully.")

    # ── Step 4: model available? ──────────────────────────────────────────────
    ensure_ollama_model(resolved_model, base_url=base_url, progress=progress)

    return resolved_model
