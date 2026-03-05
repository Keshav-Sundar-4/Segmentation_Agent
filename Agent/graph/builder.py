"""
Graph builder — assembles nodes and edges into a compiled LangGraph pipeline.

Current topology
----------------

    [planner] ──► [coder] ──► [executor] ──► END
                    ▲               │
                    └───────────────┘  (on failure, up to MAX_RETRIES)

Extending the graph
-------------------
To insert a code-reviewer between Coder and Executor:

    from .reviewer import reviewer_node          # your new node

    graph.add_node("reviewer", reviewer_node)

    # Replace the direct coder→executor edge with conditional routing:
    graph.add_conditional_edges(
        "coder",
        route_after_coder,                        # fn(state) → "reviewer"|"executor"
        {"reviewer": "reviewer", "executor": "executor"},
    )
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,                       # fn(state) → "coder"|"executor"
        {"coder": "coder", "executor": "executor"},
    )

To add a HITL interrupt before the executor:

    graph.add_node("executor", executor_node, interrupt_before=["executor"])
    # Then resume via: graph.update_state(config, {"hitl_decision": "approve"})

See LangGraph docs: https://langchain-ai.github.io/langgraph/
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from ..agents.coder import coder_node
from ..agents.planner import planner_node
from ..core.state import PipelineState
from ..tools.executor import exec_sandboxed

logger = logging.getLogger("biovision.graph")

# Maximum number of coder retries after execution failures.
MAX_RETRIES = 3


# ── Executor node ─────────────────────────────────────────────────────────────
# Kept here (not in agents/) because it is a thin orchestration wrapper around
# a tool, rather than an LLM-backed agent.


def executor_node(state: PipelineState) -> dict:
    """LangGraph node: run the generated script in a sandboxed subprocess."""
    logger.info("Executor: running generated script.")

    result = exec_sandboxed(
        code=state["generated_code"],
        input_dir=state["input_dir"],
        output_dir=state["output_dir"],
        dependencies=state.get("code_dependencies", []),
    )

    success = result["success"]

    if success:
        logger.info("Executor: script completed successfully.")
    else:
        logger.warning("Executor: script failed.\n%s", result["stderr"])

    return {
        "execution_stdout": result["stdout"],
        "execution_stderr": result["stderr"],
        "execution_success": success,
        "error": None if success else result["stderr"],
        # Increment retry counter so the router can enforce MAX_RETRIES.
        "retries": state.get("retries", 0) + (0 if success else 1),
    }


# ── Routing functions ─────────────────────────────────────────────────────────


def _route_after_executor(
    state: PipelineState,
) -> Literal["coder", "__end__"]:
    """
    After execution:
      - Success → END
      - Failure within retry budget → back to Coder (with error in state)
      - Failure beyond budget → END (Napari UI surfaces the error)
    """
    if state["execution_success"]:
        return "__end__"

    if state.get("retries", 0) < MAX_RETRIES:
        logger.info(
            "Executor: routing back to Coder (attempt %d/%d).",
            state["retries"],
            MAX_RETRIES,
        )
        return "coder"

    logger.error("Executor: max retries (%d) reached. Terminating.", MAX_RETRIES)
    return "__end__"


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
    graph.add_node("executor", executor_node)

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "executor")

    # Conditional edge: executor may loop back to coder on failure.
    graph.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"coder": "coder", "__end__": END},
    )

    compile_kwargs: dict = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = MemorySaver()

    return graph.compile(**compile_kwargs)
