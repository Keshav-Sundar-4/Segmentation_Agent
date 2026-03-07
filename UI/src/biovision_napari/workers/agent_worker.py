"""
Agent execution workers.

Two workers are provided:

run_agent_worker(command, working_dir)
    Legacy generic subprocess launcher (kept for backward compatibility).
    Launches an arbitrary shell command and streams stdout/stderr lines.

run_biovision_agent_worker(...)
    New typed worker that directly invokes the BioVision Agent Python API
    (Agent.main.run_pipeline_stream) in a background thread.
    Yields (node_name: str, state_delta: dict) tuples.
    Preferred over run_agent_worker for the agent panel.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
from pathlib import Path

from napari.qt.threading import thread_worker

_SENTINEL = object()  # signals that a reader thread is done


# ---------------------------------------------------------------------------
# Legacy subprocess worker (kept for backward compatibility)
# ---------------------------------------------------------------------------


def _pipe_reader(stream, is_stderr: bool, q: queue.Queue) -> None:
    """Read lines from a stream and put (line, is_stderr) onto the queue."""
    try:
        for line in stream:
            q.put((line.rstrip(), is_stderr))
    finally:
        q.put(_SENTINEL)


@thread_worker
def run_agent_worker(command: str, working_dir: str):
    """
    Run the agent command in a subprocess. Yields output lines as they arrive.
    Yields (line: str, is_stderr: bool) tuples.
    Raises subprocess.CalledProcessError on non-zero exit (caught by napari worker).
    """
    cwd = Path(working_dir).resolve()
    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    q: queue.Queue = queue.Queue()
    threads = [
        threading.Thread(target=_pipe_reader, args=(proc.stdout, False, q), daemon=True),
        threading.Thread(target=_pipe_reader, args=(proc.stderr, True,  q), daemon=True),
    ]
    for t in threads:
        t.start()

    # Drain the queue until both reader threads have signalled done
    sentinels_received = 0
    while sentinels_received < 2:
        item = q.get()
        if item is _SENTINEL:
            sentinels_received += 1
        else:
            yield item

    for t in threads:
        t.join()

    proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, command)


# ---------------------------------------------------------------------------
# Typed BioVision pipeline worker
# ---------------------------------------------------------------------------


def _ensure_agent_importable() -> None:
    """
    Add the repository root (the directory containing ``Agent/``) to sys.path
    if it is not already there.

    This is needed because the UI package lives in ``UI/src/`` while the
    Agent package lives at the repo root.  launch.py adds ``UI/src`` to
    sys.path but does not add the repo root.
    """
    here = Path(__file__).resolve()
    # Walk up from  UI/src/biovision_napari/workers/agent_worker.py
    for parent in here.parents:
        if (parent / "Agent" / "__init__.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return
    raise ImportError(
        "Could not locate the Agent package. "
        "Make sure BioVision is launched from the repository root."
    )


@thread_worker
def run_biovision_agent_worker(
    metadata_yaml: str,
    input_dir: str,
    output_dir: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str = "",
    llm_base_url: str = "",
):
    """
    Run the BioVision preprocessing pipeline in a background thread.

    Yields ``(node_name: str, state_delta: dict)`` tuples as each graph node
    completes — suitable for streaming progress updates into the UI.

    Parameters
    ----------
    metadata_yaml:
        Raw YAML string (contents of the metadata file, not the path).
    input_dir:
        Absolute path to the raw image folder.
    output_dir:
        Absolute path for processed output images.
    llm_provider:
        ``"anthropic"`` or ``"ollama"``.
    llm_model:
        Model identifier.  Empty → provider-specific default.
    llm_api_key:
        Anthropic API key.  Not required for Ollama.
        Never logged or persisted by this worker.
    llm_base_url:
        Ollama base URL.  Empty → ``http://localhost:11434``.

    Yields
    ------
    tuple[str, dict]
        ``(node_name, state_delta)`` — the node that just finished and the
        partial state it returned.

    Raises
    ------
    Any exception raised by the pipeline is propagated to the napari worker
    and delivered via the ``errored`` signal on the worker object.
    """
    _ensure_agent_importable()

    from Agent.main import run_pipeline_stream  # noqa: PLC0415

    for node_name, state_delta in run_pipeline_stream(
        metadata_yaml=metadata_yaml,
        input_dir=input_dir,
        output_dir=output_dir,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
    ):
        yield (node_name, state_delta)
