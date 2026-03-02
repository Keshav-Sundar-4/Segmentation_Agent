"""
hitl_worker.py — napari thread_worker bridge for the BioVision pipeline.

This worker drives PipelineRunner.run() and communicates with the Qt main
thread via napari's GeneratorWorker protocol:

  Yields (kind, payload) tuples → emitted as the `yielded` signal on main thread:
    ("log",      str)   — a log line for the UI text area
    ("review",   dict)  — mini-batch results; UI must call worker.send(decision)
    ("done",     str)   — final output directory path
    ("error",    str)   — fatal error; worker will stop after this

  Receives decisions via GeneratorWorker.send(value):
    After yielding ("review", …), the worker suspends.
    The UI calls worker.send({"action": "accept"|"reject", "feedback": str})
    to resume it and forward the decision into PipelineRunner.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from napari.qt.threading import thread_worker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent directory bootstrap — ensures Agent/ is importable regardless of how
# the UI package was installed.
# ---------------------------------------------------------------------------


def _ensure_agent_on_path() -> bool:
    """
    Walk up from this file's location to find the repo's Agent/ directory and
    prepend it to sys.path so that `from core.pipeline import PipelineRunner`
    succeeds.  Returns True on success.
    """
    start = Path(__file__).resolve()
    for parent in [start, *start.parents]:
        candidate = parent / "Agent"
        if candidate.is_dir() and (candidate / "core" / "pipeline.py").exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return True
    return False


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


@thread_worker
def hitl_worker(
    input_folder: str,
    metadata_path: str,
    api_key: str,
    sample_size: int = 5,
    output_root: str = "outputs",
):
    """
    Background GeneratorWorker that runs the BioVision pipeline agent.

    Parameters
    ----------
    input_folder  : path to the directory containing input images
    metadata_path : path to the metadata YAML file
    api_key       : Anthropic API key (never logged)
    sample_size   : number of images for the mini-batch preview (default 5)
    output_root   : root directory for all outputs

    Yields
    ------
    (kind, payload) tuples — see module docstring.
    """
    # ── Bootstrap import ───────────────────────────────────────────────────
    if not _ensure_agent_on_path():
        yield ("error", "Cannot locate Agent/core/pipeline.py. Check repository layout.")
        return

    try:
        from core.pipeline import PipelineRunner                    # noqa: PLC0415
        from agents.preprocessing.agent import PreprocessingAgent   # noqa: PLC0415
    except ImportError as exc:
        yield ("error", f"Failed to import pipeline modules: {exc}")
        return

    # ── Read metadata ──────────────────────────────────────────────────────
    yield ("log", "Reading metadata file…")
    try:
        metadata_content = Path(metadata_path).read_text(encoding="utf-8")
    except Exception as exc:
        yield ("error", f"Cannot read metadata file: {exc}")
        return

    # ── Prepare output root ────────────────────────────────────────────────
    out_root = Path(output_root)
    try:
        out_root.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        yield ("error", f"Cannot create output directory '{out_root}': {exc}")
        return

    # ── Build runner ───────────────────────────────────────────────────────
    yield ("log", "Initialising pipeline agent…")
    try:
        runner = (
            PipelineRunner(
                api_key=api_key,
                sample_size=sample_size,
                output_root=out_root,
            )
            .add_agent(PreprocessingAgent())
            .with_hitl()
            .build()
        )
    except Exception as exc:
        yield ("error", f"Failed to initialise agent: {exc}")
        return

    gen = runner.run(input_folder=input_folder, metadata_content=metadata_content)

    # ── Drive the generator ────────────────────────────────────────────────
    # value_to_send is None on the first call (use next()), then holds the
    # human decision that must be forwarded back to the runner via gen.send().
    value_to_send = None

    while True:
        try:
            if value_to_send is None:
                event = next(gen)
            else:
                event = gen.send(value_to_send)
                value_to_send = None
        except StopIteration:
            break
        except Exception as exc:
            yield ("error", f"Agent error: {exc}")
            logger.exception("Unhandled error in pipeline agent")
            return

        kind, payload = event

        if kind in ("log", "error"):
            yield (kind, payload)
            if kind == "error":
                return

        elif kind == "review":
            # Suspend this worker and hand control back to the UI.
            # The UI must call worker.send({"action": …, "feedback": …}).
            decision = yield ("review", payload)

            if decision is None:
                # Worker was stopped (e.g. Stop button) while awaiting review
                return

            # Forward decision to the PipelineRunner on next iteration
            value_to_send = decision

        elif kind == "done":
            yield ("done", payload)
            return

        else:
            yield ("log", f"[unknown event '{kind}'] {payload}")
