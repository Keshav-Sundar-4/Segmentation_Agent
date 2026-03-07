"""
Graph builder — assembles nodes and edges into a compiled LangGraph pipeline.

Current topology
----------------

    [planner] ──► [coder] ──► [sandbox_executor] ──► [local_executor] ──► END
                    ▲                │
                    └────────────────┘  (on failure, up to MAX_RETRIES)

Two-stage execution
-------------------
1. sandbox_executor  — Docker back-end, runs against the small ``sample_dir``.
   Validates that the generated code and its dependencies actually work before
   touching the full image dataset.  Failures (pip errors, runtime crashes)
   route back to the Coder with the error in state.

2. local_executor    — Subprocess back-end, runs against the full ``input_dir``.
   Only reached once the sandbox stage succeeds.  Routes to END regardless of
   outcome (errors are surfaced to the caller via the state).

Extending the graph
-------------------
To add a HITL interrupt before the sandbox:

    graph.add_node("sandbox_executor", sandbox_executor_node,
                   interrupt_before=["sandbox_executor"])

See LangGraph docs: https://langchain-ai.github.io/langgraph/
"""

from __future__ import annotations

import logging
import tempfile
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..agents.coder import coder_node
from ..agents.planner import planner_node
from ..core.state import PipelineState
from ..tools.executor import exec_sandboxed

logger = logging.getLogger("biovision.graph")

# Maximum number of coder retries after sandbox execution failures.
MAX_RETRIES = 3


# ── Executor nodes ────────────────────────────────────────────────────────────


def sandbox_executor_node(state: PipelineState) -> dict:
    """
    LangGraph node: validate the generated script inside Docker using sample_dir.

    Uses the Docker back-end so the code runs in a fully isolated container.
    Succeeds fast on 1-2 sample images; failures are cheap to retry.
    """
    logger.info("SandboxExecutor: running generated script against sample_dir.")

    # Create a throwaway output dir for sandbox results (discarded after).
    with tempfile.TemporaryDirectory(prefix="biovision_sandbox_out_") as sandbox_out:
        result = exec_sandboxed(
            code=state["generated_code"],
            input_dir=state["sample_dir"],
            output_dir=sandbox_out,
            dependencies=state.get("code_dependencies", []),
            use_docker=True,
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


# ── Routing functions ─────────────────────────────────────────────────────────


def _route_after_sandbox(
    state: PipelineState,
) -> Literal["coder", "local_executor"]:
    """
    After sandbox execution:
      - Success               → local_executor (run against full dataset)
      - Failure within budget → coder (LLM self-corrects with the error)
      - Failure beyond budget → local_executor with execution_success=False
                                so the caller can surface the error cleanly
    """
    if state["execution_success"]:
        return "local_executor"

    if state.get("retries", 0) < MAX_RETRIES:
        logger.info(
            "SandboxExecutor: routing back to Coder (attempt %d/%d).",
            state["retries"],
            MAX_RETRIES,
        )
        return "coder"

    logger.error(
        "SandboxExecutor: max retries (%d) reached. Passing failure to local_executor.",
        MAX_RETRIES,
    )
    # Terminate via local_executor so the graph always exits through that node
    # and the final state has a consistent shape for the caller.
    return "local_executor"


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
    graph.add_node("planner", planner_node)
    graph.add_node("coder", coder_node)
    graph.add_node("sandbox_executor", sandbox_executor_node)
    graph.add_node("local_executor", local_executor_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "sandbox_executor")

    # Conditional edge: sandbox may loop back to coder on failure.
    graph.add_conditional_edges(
        "sandbox_executor",
        _route_after_sandbox,
        {"coder": "coder", "local_executor": "local_executor"},
    )

    # local_executor always terminates the graph.
    graph.add_edge("local_executor", END)

    compile_kwargs: dict = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = MemorySaver()

    return graph.compile(**compile_kwargs)
