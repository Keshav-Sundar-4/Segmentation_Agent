"""
BioVision agent — public entry point.

Called by the Napari UI worker or directly from the command line for
development and testing.

Minimal usage
-------------
    from Agent.main import run_pipeline

    result = run_pipeline(
        metadata_yaml=open("meta.yaml").read(),
        input_dir="/data/raw",
        output_dir="/data/processed",
    )
    print(result["execution_success"])   # True / False
    print(result["plan_title"])          # e.g. "CLAHE + Gaussian Denoising"

Streaming (progress updates)
-----------------------------
    for event in run_pipeline_stream(metadata_yaml, input_dir, output_dir):
        node_name, state_snapshot = event
        print(f"[{node_name}] done")

Environment variables
---------------------
    ANTHROPIC_API_KEY   — required if api_key= argument is not passed
    BIOVISION_USE_DOCKER — set to "1" to use Docker execution back-end
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("biovision")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "An Anthropic API key is required. "
            "Pass api_key= or set the ANTHROPIC_API_KEY environment variable."
        )
    return key


def _make_initial_state(
    metadata_yaml: str,
    input_dir: str,
    output_dir: str,
    api_key: str,
) -> dict:
    """Return a fully-initialised PipelineState dict (all fields populated)."""
    return {
        # ── Inputs ────────────────────────────────────────────────────────────
        "metadata_yaml": metadata_yaml,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "api_key": api_key,
        # ── Planner outputs (populated by planner_node) ───────────────────────
        "plan_title": "",
        "plan_steps": [],
        "plan_rationale": "",
        # ── Coder outputs (populated by coder_node) ───────────────────────────
        "generated_code": "",
        "code_dependencies": [],
        # ── Executor outputs (populated by executor_node) ─────────────────────
        "execution_stdout": "",
        "execution_stderr": "",
        "execution_success": False,
        # ── Control ───────────────────────────────────────────────────────────
        "error": None,
        "retries": 0,
        "messages": [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    metadata_yaml: str,
    input_dir: str,
    output_dir: str,
    api_key: Optional[str] = None,
    *,
    thread_id: str = "default",
    checkpointer: bool = False,
) -> dict:
    """
    Execute the full preprocessing pipeline and return the final state.

    Parameters
    ----------
    metadata_yaml:
        Raw YAML string describing the dataset (passed from the Napari UI).
    input_dir:
        Absolute path to the folder containing raw images.
    output_dir:
        Absolute path where processed images will be saved.
    api_key:
        Anthropic API key.  Falls back to ANTHROPIC_API_KEY env-var.
    thread_id:
        Checkpointer thread identifier.  Use a unique value per run when
        ``checkpointer=True`` to keep state isolated between runs.
    checkpointer:
        Attach an in-memory MemorySaver for HITL / pause-resume support.

    Returns
    -------
    dict
        Final ``PipelineState`` snapshot after the graph reaches END.
    """
    from .graph.builder import build_graph  # deferred import for fast startup

    key = _resolve_key(api_key)
    initial = _make_initial_state(metadata_yaml, input_dir, output_dir, key)

    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    final: dict = graph.invoke(initial, config=config)

    logger.info(
        "Pipeline complete — success=%s, plan='%s'.",
        final.get("execution_success"),
        final.get("plan_title"),
    )
    return final


def run_pipeline_stream(
    metadata_yaml: str,
    input_dir: str,
    output_dir: str,
    api_key: Optional[str] = None,
    *,
    thread_id: str = "default",
    checkpointer: bool = False,
) -> Iterator[tuple[str, dict]]:
    """
    Stream node-by-node state snapshots as the pipeline executes.

    Yields ``(node_name, state_snapshot)`` tuples.  Useful for the Napari
    worker thread to surface progress updates in the UI without blocking.

    Example
    -------
        for node, snapshot in run_pipeline_stream(yaml, in_dir, out_dir):
            print(f"[{node}] plan_title={snapshot.get('plan_title', '…')}")
    """
    from .graph.builder import build_graph

    key = _resolve_key(api_key)
    initial = _make_initial_state(metadata_yaml, input_dir, output_dir, key)

    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}

    for chunk in graph.stream(initial, config=config):
        # chunk is {node_name: state_delta}
        for node_name, state_delta in chunk.items():
            yield node_name, state_delta


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioVision image preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("metadata_yaml", help="Path to a metadata YAML file")
    parser.add_argument("input_dir", help="Directory containing raw images")
    parser.add_argument("output_dir", help="Directory to write processed images")
    args = parser.parse_args()

    with open(args.metadata_yaml, encoding="utf-8") as fh:
        yaml_str = fh.read()

    result = run_pipeline(yaml_str, args.input_dir, args.output_dir)

    print("\n── Pipeline summary ──────────────────────────────────────")
    print(f"  Plan      : {result['plan_title']}")
    print(f"  Steps     : {len(result['plan_steps'])}")
    print(f"  Success   : {result['execution_success']}")
    print(f"  Retries   : {result['retries']}")
    if result["execution_stdout"]:
        print(f"\nstdout:\n{result['execution_stdout']}")
    if result["execution_stderr"]:
        print(f"\nstderr:\n{result['execution_stderr']}")
