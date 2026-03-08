"""
Graph builder — assembles nodes and edges into a compiled LangGraph pipeline.

Current topology
----------------

    [planner] ──► [coder] ──► [sandbox_executor] ──► [local_executor] ──► END
                    ▲                │
                    └────────────────┘  (on failure, up to MAX_RETRIES)
                                    │
                                    └──► [terminal_failure] ──► END
                                         (when retries exhausted — full dataset
                                          is NEVER touched on validation failure)

Two-stage execution
-------------------
1. sandbox_executor  — Subprocess back-end (Docker optional via
   BIOVISION_USE_DOCKER=1), runs against the small ``sample_dir``.
   Validates that the generated code and its dependencies actually work before
   touching the full image dataset.  Failures (pip errors, runtime crashes)
   route back to the Coder with the error in state, up to MAX_RETRIES times.
   After MAX_RETRIES, routes to terminal_failure — the full dataset is NOT
   processed.

2. local_executor    — Subprocess back-end, runs against the full ``input_dir``.
   Only reached once the sandbox stage succeeds.  Routes to END regardless of
   outcome (errors are surfaced to the caller via the state).

Docker support
--------------
Sandbox execution uses subprocess by default for minimal friction.
Set the environment variable BIOVISION_USE_DOCKER=1 to use Docker instead.
If Docker is requested but the docker binary is not found, the error is
surfaced in execution_stderr and the node fails cleanly (triggering a retry
or terminal_failure as usual).

Extending the graph
-------------------
To add a HITL interrupt before the sandbox:

    graph.add_node("sandbox_executor", sandbox_executor_node,
                   interrupt_before=["sandbox_executor"])

See LangGraph docs: https://langchain-ai.github.io/langgraph/
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..agents.coder import coder_node
from ..agents.planner import planner_node
from ..core.state import PipelineState
from ..tools.executor import exec_sandboxed

logger = logging.getLogger("biovision.graph")

# Default maximum number of coder retries after sandbox execution failures.
# Overridden at runtime by state["max_retries"] if set.
MAX_RETRIES = 3

# Use Docker for sandbox execution when this env-var is set to "1".
# Defaults to False so basic local usage works without Docker.
_USE_DOCKER: bool = os.environ.get("BIOVISION_USE_DOCKER", "0") == "1"


# ── Executor nodes ────────────────────────────────────────────────────────────


def sandbox_executor_node(state: PipelineState) -> dict:
    """
    LangGraph node: validate the generated script against sample_dir.

    Uses subprocess by default (Docker when BIOVISION_USE_DOCKER=1).
    Succeeds fast on 1-2 sample images; failures are cheap to retry.
    """
    logger.info(
        "SandboxExecutor: running generated script against sample_dir "
        "(docker=%s).", _USE_DOCKER
    )

    # Throwaway output dir for sandbox results — discarded after validation.
    with tempfile.TemporaryDirectory(prefix="biovision_sandbox_out_") as sandbox_out:
        result = exec_sandboxed(
            code=state["generated_code"],
            input_dir=state["sample_dir"],
            output_dir=sandbox_out,
            dependencies=state.get("code_dependencies", []),
            use_docker=_USE_DOCKER,
        )

    success = result["success"]

    if success:
        logger.info("SandboxExecutor: script completed successfully.")
    else:
        logger.warning("SandboxExecutor: script failed.\n%s", result["stderr"])

    return {
        "execution_stdout": result["stdout"],
        "execution_stderr": result["stderr"],
        "execution_success": success,
        "error": None if success else result["stderr"],
        # On success, lock in the validated dependency list for local_executor.
        "validated_dependencies": state.get("code_dependencies", []) if success else [],
        # Increment retry counter on failure so the router can enforce MAX_RETRIES.
        "retries": state.get("retries", 0) + (0 if success else 1),
    }


def local_executor_node(state: PipelineState) -> dict:
    """
    LangGraph node: run the validated script natively against the full input_dir.

    Uses the subprocess back-end (no Docker overhead) because the code has
    already been proven correct in the sandbox stage.  Uses ``validated_dependencies``
    (the dep list confirmed to install in the sandbox) to avoid re-resolving.
    """
    logger.info("LocalExecutor: running validated script against full input_dir.")

    deps = state.get("validated_dependencies") or state.get("code_dependencies", [])

    result = exec_sandboxed(
        code=state["generated_code"],
        input_dir=state["input_dir"],
        output_dir=state["output_dir"],
        dependencies=deps,
        use_docker=False,
    )

    success = result["success"]

    if success:
        logger.info("LocalExecutor: script completed successfully.")
    else:
        logger.error("LocalExecutor: script failed.\n%s", result["stderr"])

    return {
        "execution_stdout": result["stdout"],
        "execution_stderr": result["stderr"],
        "execution_success": success,
        "error": None if success else result["stderr"],
    }


def terminal_failure_node(state: PipelineState) -> dict:
    """
    LangGraph node: sandbox retries exhausted — terminate without running full dataset.

    This node is reached ONLY when MAX_RETRIES is exhausted and the sandbox
    validation still fails.  The full image dataset is NEVER touched.
    """
    msg = (
        state.get("error")
        or "Sandbox validation failed after maximum retries. Full dataset was not processed."
    )
    logger.error("Pipeline: terminal failure — %s", msg)
    return {
        "execution_success": False,
        "error": msg,
    }


# ── Routing functions ─────────────────────────────────────────────────────────


def _route_after_planner(
    state: PipelineState,
) -> Literal["coder", "terminal_failure"]:
    """After planning: if the planner recorded an error, abort immediately."""
    if state.get("error"):
        logger.error("Planner failed — routing to terminal_failure.")
        return "terminal_failure"
    return "coder"


def _route_after_coder(
    state: PipelineState,
) -> Literal["sandbox_executor", "terminal_failure"]:
    """After coding: if the coder recorded an error, abort immediately."""
    if state.get("error"):
        logger.error("Coder failed — routing to terminal_failure.")
        return "terminal_failure"
    return "sandbox_executor"


def _route_after_sandbox(
    state: PipelineState,
) -> Literal["coder", "local_executor", "terminal_failure"]:
    """
    After sandbox execution:
      - Success               → local_executor  (run against full dataset)
      - Failure within budget → coder           (LLM self-corrects with error)
      - Failure beyond budget → terminal_failure (clean abort; full dataset untouched)
    """
    if state["execution_success"]:
        return "local_executor"

    retries = state.get("retries", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)
    if retries < max_retries:
        logger.info(
            "SandboxExecutor: routing back to Coder (attempt %d/%d).",
            retries,
            max_retries,
        )
        return "coder"

    logger.error(
        "SandboxExecutor: max retries (%d) reached. Terminating — full dataset untouched.",
        max_retries,
    )
    return "terminal_failure"


# ── Public factory ────────────────────────────────────────────────────────────


def build_graph(*, checkpointer: bool = False):
    """
    Build and compile the preprocessing pipeline graph.

    Parameters
    ----------
    checkpointer:
        Attach an in-memory MemorySaver when True.  Required for:
          - pause / resume (interrupt_before / interrupt_after)
          - HITL gates (graph.update_state + graph.invoke with same thread_id)
          - graph.get_state() introspection

    Returns
    -------
    CompiledStateGraph
        Call `.invoke(initial_state, config)` or `.stream(...)` on the result.
    """
    graph: StateGraph = StateGraph(PipelineState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("planner",          planner_node)
    graph.add_node("coder",            coder_node)
    graph.add_node("sandbox_executor", sandbox_executor_node)
    graph.add_node("local_executor",   local_executor_node)
    graph.add_node("terminal_failure", terminal_failure_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.set_entry_point("planner")

    # Planner may fail (LLM error, bad structured output) → terminal_failure
    graph.add_conditional_edges(
        "planner",
        _route_after_planner,
        {"coder": "coder", "terminal_failure": "terminal_failure"},
    )

    # Coder may fail → terminal_failure
    graph.add_conditional_edges(
        "coder",
        _route_after_coder,
        {"sandbox_executor": "sandbox_executor", "terminal_failure": "terminal_failure"},
    )

    # Conditional: sandbox may loop back to coder or abort to terminal_failure.
    graph.add_conditional_edges(
        "sandbox_executor",
        _route_after_sandbox,
        {
            "coder":            "coder",
            "local_executor":   "local_executor",
            "terminal_failure": "terminal_failure",
        },
    )

    # Both terminal paths go to END.
    graph.add_edge("local_executor",   END)
    graph.add_edge("terminal_failure", END)

    compile_kwargs: dict = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = MemorySaver()

    return graph.compile(**compile_kwargs)
