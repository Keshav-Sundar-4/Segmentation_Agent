"""
BioVision agent — public entry point.

Called by the Napari UI worker or directly from the command line for
development and testing.

Minimal usage (Anthropic / Claude)
-----------------------------------
    from Agent.main import run_pipeline

    result = run_pipeline(
        metadata_yaml=open("meta.yaml").read(),
        input_dir="/data/raw",
        api_key="sk-ant-…",          # or set ANTHROPIC_API_KEY env-var
    )
    print(result["execution_success"])   # True / False
    print(result["plan_title"])          # e.g. "CLAHE + Gaussian Denoising"

Minimal usage (Ollama / local)
-------------------------------
    result = run_pipeline(
        metadata_yaml=open("meta.yaml").read(),
        input_dir="/data/raw",
        llm_provider="ollama",
        llm_model="llama3.2",
    )

Streaming (progress updates)
-----------------------------
    for node_name, state_delta in run_pipeline_stream(
        metadata_yaml, input_dir, llm_provider="anthropic", api_key="sk-ant-…"
    ):
        print(f"[{node_name}] done")

Provider / model arguments
--------------------------
    llm_provider  : "anthropic" (default) | "ollama"
    llm_model     : model identifier; empty → role-specific default
    llm_api_key   : Anthropic API key (or use env ANTHROPIC_API_KEY)
    llm_base_url  : Ollama base URL (default: http://localhost:11434)

    # Backward-compat alias — treated the same as llm_api_key for Anthropic:
    api_key       : Anthropic API key

Environment variables
---------------------
    ANTHROPIC_API_KEY    — Anthropic key fallback when llm_api_key / api_key not passed
    BIOVISION_USE_DOCKER — set to "1" to use Docker for sandbox execution

Auto-sampling
-------------
    When ``input_dir`` is provided, the pipeline automatically creates a
    temporary ``sample_dir`` containing 1-2 randomly chosen images.  The
    sandbox executor validates generated code against these samples before
    running on the full dataset.  The temporary directory is cleaned up
    automatically after the pipeline finishes.
"""

from __future__ import annotations

import logging
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("biovision")

# Image extensions recognised when sampling from input_dir.
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ndpi", ".svs"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_anthropic_key(llm_api_key: Optional[str], api_key: Optional[str]) -> str:
    """Return the Anthropic API key, checking both args then the env-var."""
    key = llm_api_key or api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "An Anthropic API key is required. "
            "Pass llm_api_key= (or api_key=) or set the ANTHROPIC_API_KEY "
            "environment variable."
        )
    return key


def _create_sample_dir(input_dir: str) -> str:
    """
    Create a temporary directory containing 1-2 randomly selected images from
    *input_dir*.  The caller is responsible for deleting the directory when done.

    Raises
    ------
    ValueError
        If no image files are found in *input_dir*.
    """
    input_path = Path(input_dir)
    images = [
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(
            f"No image files found in {input_dir!r}. "
            f"Supported extensions: {sorted(_IMAGE_EXTENSIONS)}"
        )

    sample = random.sample(images, min(2, len(images)))
    tmp_dir = tempfile.mkdtemp(prefix="biovision_sample_")
    for img in sample:
        shutil.copy2(img, tmp_dir)

    logger.info(
        "Auto-sampling: created sample_dir=%r with %d image(s): %s",
        tmp_dir,
        len(sample),
        [p.name for p in sample],
    )
    return tmp_dir


def _make_initial_state(
    metadata_yaml: str,
    input_dir: str,
    output_dir: str,
    sample_dir: str,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    # Kept for backward-compat callers that inspect api_key directly.
    api_key: str = "",
) -> dict:
    """Return a fully-initialised PipelineState dict (all fields populated)."""
    return {
        # ── Inputs ────────────────────────────────────────────────────────────
        "metadata_yaml": metadata_yaml,
        "input_dir":     input_dir,
        "output_dir":    output_dir,
        "sample_dir":    sample_dir,
        # ── LLM runtime configuration ─────────────────────────────────────────
        "llm_provider": llm_provider,
        "llm_model":    llm_model,
        "llm_api_key":  llm_api_key,
        "llm_base_url": llm_base_url,
        # Deprecated alias — kept so any code still reading state["api_key"] works.
        "api_key": api_key or llm_api_key,
        # ── Planner outputs (populated by planner_node) ───────────────────────
        "plan_title":    "",
        "plan_steps":    [],
        "plan_rationale": "",
        # ── Coder outputs (populated by coder_node) ───────────────────────────
        "generated_code":    "",
        "code_dependencies": [],
        # ── Executor outputs (populated by executor nodes) ────────────────────
        "execution_stdout":       "",
        "execution_stderr":       "",
        "execution_success":      False,
        "validated_dependencies": [],
        # ── Control ───────────────────────────────────────────────────────────
        "error":       None,
        "retries":     0,
        "max_retries": 3,
        "messages":    [],
    }


def _prepare_run(
    metadata_yaml: str,
    input_dir: str,
    output_dir: Optional[str],
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    api_key: str,
) -> tuple[dict, str]:
    """
    Validate inputs, resolve defaults, create sample dir, and return
    ``(initial_state, sample_dir)`` so the caller can always clean up.
    """
    resolved_output = output_dir or str(
        Path(input_dir).parent / (Path(input_dir).name + "_biovision_output")
    )
    sample_dir = _create_sample_dir(input_dir)
    initial = _make_initial_state(
        metadata_yaml=metadata_yaml,
        input_dir=input_dir,
        output_dir=resolved_output,
        sample_dir=sample_dir,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        api_key=api_key,
    )
    return initial, sample_dir


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_pipeline(
    metadata_yaml: str,
    input_dir: str,
    output_dir: Optional[str] = None,
    # ── Anthropic backward-compat alias ───────────────────────────────────────
    api_key: Optional[str] = None,
    *,
    # ── Provider / model settings ─────────────────────────────────────────────
    llm_provider: str = "anthropic",
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    # ── Graph options ─────────────────────────────────────────────────────────
    thread_id: str = "default",
    checkpointer: bool = False,
    max_retries: int = 3,
) -> dict:
    """
    Execute the full preprocessing pipeline and return the final state.

    Parameters
    ----------
    metadata_yaml:
        Raw YAML string describing the dataset.
    input_dir:
        Absolute path to the folder containing raw images.
    output_dir:
        Absolute path where processed images will be saved.
        Defaults to ``<input_dir>_biovision_output``.
    api_key:
        Anthropic API key (backward-compat alias for ``llm_api_key``).
        Falls back to the ANTHROPIC_API_KEY environment variable.
    llm_provider:
        ``"anthropic"`` (default) or ``"ollama"``.
    llm_model:
        Model identifier.  Empty → role-specific default for the provider.
    llm_api_key:
        Anthropic API key.  Falls back to ``api_key`` then ANTHROPIC_API_KEY.
    llm_base_url:
        Ollama base URL.  Empty → ``http://localhost:11434``.
    thread_id:
        Checkpointer thread identifier.
    checkpointer:
        Attach an in-memory MemorySaver for HITL / pause-resume support.

    Returns
    -------
    dict
        Final ``PipelineState`` snapshot after the graph reaches END.
    """
    from .graph.builder import build_graph  # deferred import for fast startup

    resolved_provider = (llm_provider or "anthropic").lower().strip()

    # Resolve the Anthropic key — only required for the anthropic provider.
    resolved_api_key = ""
    if resolved_provider == "anthropic":
        resolved_api_key = _resolve_anthropic_key(llm_api_key, api_key)

    initial, sample_dir = _prepare_run(
        metadata_yaml=metadata_yaml,
        input_dir=input_dir,
        output_dir=output_dir,
        llm_provider=resolved_provider,
        llm_model=llm_model or "",
        llm_api_key=resolved_api_key,
        llm_base_url=llm_base_url or "",
        api_key=api_key or "",
    )
    initial["max_retries"] = max_retries

    try:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        final: dict = graph.invoke(initial, config=config)
    finally:
        shutil.rmtree(sample_dir, ignore_errors=True)
        logger.info("Auto-sampling: cleaned up sample_dir=%r.", sample_dir)

    logger.info(
        "Pipeline complete — success=%s, plan='%s'.",
        final.get("execution_success"),
        final.get("plan_title"),
    )
    return final


def run_pipeline_stream(
    metadata_yaml: str,
    input_dir: str,
    output_dir: Optional[str] = None,
    # ── Anthropic backward-compat alias ───────────────────────────────────────
    api_key: Optional[str] = None,
    *,
    # ── Provider / model settings ─────────────────────────────────────────────
    llm_provider: str = "anthropic",
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    # ── Graph options ─────────────────────────────────────────────────────────
    thread_id: str = "default",
    checkpointer: bool = False,
    max_retries: int = 3,
    stream_tokens: bool = True,
) -> Iterator[tuple[str, dict]]:
    """
    Stream node-by-node state snapshots as the pipeline executes.

    Yields ``(node_name, state_delta)`` tuples for node completions, plus
    ``("_token", {"node": str, "token": str})`` tuples for individual LLM
    output tokens when ``stream_tokens=True`` (the default).

    See ``run_pipeline`` for parameter documentation.
    """
    from .graph.builder import build_graph

    resolved_provider = (llm_provider or "anthropic").lower().strip()

    resolved_api_key = ""
    if resolved_provider == "anthropic":
        resolved_api_key = _resolve_anthropic_key(llm_api_key, api_key)

    initial, sample_dir = _prepare_run(
        metadata_yaml=metadata_yaml,
        input_dir=input_dir,
        output_dir=output_dir,
        llm_provider=resolved_provider,
        llm_model=llm_model or "",
        llm_api_key=resolved_api_key,
        llm_base_url=llm_base_url or "",
        api_key=api_key or "",
    )
    initial["max_retries"] = max_retries

    try:
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        if stream_tokens:
            # stream_mode list yields (mode, data) tuples:
            #   "messages" → (AIMessageChunk, metadata_dict)
            #   "updates"  → {node_name: state_delta}
            for mode, data in graph.stream(
                initial, config=config,
                stream_mode=["updates", "messages"],
            ):
                if mode == "messages":
                    msg_chunk, metadata = data
                    node = metadata.get("langgraph_node", "")
                    # Extract text content — str for most models,
                    # list of content blocks for Anthropic tool-use.
                    content = ""
                    raw = getattr(msg_chunk, "content", "")
                    if isinstance(raw, str):
                        content = raw
                    elif isinstance(raw, list):
                        for block in raw:
                            if isinstance(block, dict):
                                content += block.get("partial_json", "")
                                content += block.get("text", "")
                    if content and node:
                        yield "_token", {"node": node, "token": content}
                elif mode == "updates":
                    for node_name, state_delta in data.items():
                        yield node_name, state_delta
        else:
            for chunk in graph.stream(initial, config=config):
                for node_name, state_delta in chunk.items():
                    yield node_name, state_delta
    finally:
        shutil.rmtree(sample_dir, ignore_errors=True)
        logger.info("Auto-sampling: cleaned up sample_dir=%r.", sample_dir)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BioVision image preprocessing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Anthropic (key from env-var)
  python -m Agent.main meta.yaml /data/raw

  # Anthropic with explicit key
  python -m Agent.main meta.yaml /data/raw --api-key sk-ant-…

  # Ollama / local
  python -m Agent.main meta.yaml /data/raw --provider ollama --model llama3.2
""",
    )
    parser.add_argument("metadata_yaml", help="Path to a metadata YAML file")
    parser.add_argument("input_dir",     help="Directory containing raw images")
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write processed images (default: <input_dir>_biovision_output)",
    )
    parser.add_argument(
        "--provider", default="anthropic",
        choices=["anthropic", "ollama"],
        help="LLM provider (default: anthropic)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Model identifier (default: provider-specific default)",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env-var)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Ollama base URL (default: http://localhost:11434)",
    )
    args = parser.parse_args()

    with open(args.metadata_yaml, encoding="utf-8") as fh:
        yaml_str = fh.read()

    result = run_pipeline(
        yaml_str,
        args.input_dir,
        args.output_dir,
        llm_provider=args.provider,
        llm_model=args.model,
        llm_api_key=args.api_key,
        llm_base_url=args.base_url,
    )

    print("\n── Pipeline summary ──────────────────────────────────────")
    print(f"  Plan      : {result['plan_title']}")
    print(f"  Steps     : {len(result['plan_steps'])}")
    print(f"  Success   : {result['execution_success']}")
    print(f"  Retries   : {result['retries']}")
    if result["execution_stdout"]:
        print(f"\nstdout:\n{result['execution_stdout']}")
    if result["execution_stderr"]:
        print(f"\nstderr:\n{result['execution_stderr']}")
