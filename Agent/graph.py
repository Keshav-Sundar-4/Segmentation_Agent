"""
graph.py — The Brain/Manager of the BioVision HITL pipeline.

This file owns three things:
  1. Edge routing functions  — decide which node to run next
  2. build_graph()           — assembles the StateGraph from imported nodes
  3. HITLRunner              — compiles the graph and exposes it to the napari UI

The actual work (LLM calls, REPL execution, etc.) lives in nodes.py.
The shared state schema lives in state.py.

════════════════════════════════════════════════════════════════════════════════
HOW TO ADD A NEW NODE / AGENT TO THE GRAPH
════════════════════════════════════════════════════════════════════════════════

Step 1 — Write the node function in nodes.py (see the guide at the top of that
         file for the exact pattern and docstring conventions).

Step 2 — Import it here:
           from nodes import ..., your_new_node

Step 3 — Register it:
           g.add_node("your_new_node", your_new_node)

Step 4 — Wire its incoming edge(s):
           # Simple unconditional edge — always runs after some_previous_node:
           g.add_edge("some_previous_node", "your_new_node")

           # Conditional edge — runs only when a routing function says so:
           def _route_after_something(state: HITLState) -> str:
               return "your_new_node" if state["some_flag"] else "other_node"
           g.add_conditional_edges("some_previous_node", _route_after_something)

Step 5 — Wire its outgoing edge(s) the same way.

Example — adding a "quality_control" node between execute_mini_batch and
           human_review:

    # In build_graph(), replace:
    #   g.add_edge("execute_mini_batch", "human_review")
    # with:
    g.add_node("quality_control", quality_control_node)
    g.add_edge("execute_mini_batch", "quality_control")
    g.add_edge("quality_control",    "human_review")

Nothing else needs to change.  HITLRunner.run() will automatically stream
log messages from the new node's status_log output.
════════════════════════════════════════════════════════════════════════════════

Graph topology
──────────────
  START
    └─► load_metadata
          └─► research_technique
                └─► generate_code          (≤3 internal retry attempts)
                      ├─► [END]             ← code gen failed completely
                      └─► execute_mini_batch
                              └─► human_review   ◄── interrupt() — pauses here
                                      ├── accept ──► execute_full_dataset ──► END
                                      └── reject ──► process_rejection
                                                          ├─► [END]   ← 3 rejections
                                                          └─► research_technique  (loop)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from state import HITLState, MAX_REJECTIONS
from nodes import (
    execute_full_dataset_node,
    execute_mini_batch_node,
    generate_code_node,
    human_review_node,
    load_metadata_node,
    process_rejection_node,
    research_technique_node,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Edge routing functions
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_generate(state: HITLState) -> str:
    """Skip the preview if code generation failed entirely."""
    if state.get("last_error") and not state.get("technique_code"):
        return END
    return "execute_mini_batch"


def _route_after_review(state: HITLState) -> str:
    return "execute_full_dataset" if state.get("human_decision") == "accept" else "process_rejection"


def _route_after_rejection(state: HITLState) -> str:
    if state.get("rejection_count", 0) >= MAX_REJECTIONS:
        return END
    return "research_technique"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Assemble and return the uncompiled StateGraph.

    Call .compile(checkpointer=MemorySaver()) on the result before streaming.
    HITLRunner does this automatically; you only need to call build_graph()
    directly if you want to inspect or visualise the graph structure.
    """
    g = StateGraph(HITLState)

    # ── Register nodes ────────────────────────────────────────────────────
    g.add_node("load_metadata",        load_metadata_node)
    g.add_node("research_technique",   research_technique_node)
    g.add_node("generate_code",        generate_code_node)
    g.add_node("execute_mini_batch",   execute_mini_batch_node)
    g.add_node("human_review",         human_review_node)
    g.add_node("process_rejection",    process_rejection_node)
    g.add_node("execute_full_dataset", execute_full_dataset_node)

    # ── Wire edges ────────────────────────────────────────────────────────
    g.add_edge(START,                  "load_metadata")
    g.add_edge("load_metadata",        "research_technique")
    g.add_edge("research_technique",   "generate_code")
    g.add_conditional_edges("generate_code",     _route_after_generate)
    g.add_edge("execute_mini_batch",   "human_review")
    g.add_conditional_edges("human_review",      _route_after_review)
    g.add_conditional_edges("process_rejection", _route_after_rejection)
    g.add_edge("execute_full_dataset", END)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# HITLRunner — the public interface consumed by the napari worker
# ─────────────────────────────────────────────────────────────────────────────

class HITLRunner:
    """
    Compiles the LangGraph graph and exposes a generator-based interface that
    the napari hitl_worker can iterate and interact with.

    The generator protocol
    ──────────────────────
    This class deliberately uses Python's native generator mechanics so the
    napari Qt main thread never blocks:

      gen = runner.run(folder, metadata)

      # Drive the generator from the napari thread_worker:
      value_to_send = None
      while True:
          try:
              event = next(gen) if value_to_send is None else gen.send(value_to_send)
              value_to_send = None
          except StopIteration:
              break

          kind, payload = event

          if kind == "log":
              # Append payload (str) to the UI log widget
              pass

          elif kind == "review":
              # payload = {technique_name, technique_description, mini_batch_results}
              # Display in UI, then collect the human decision:
              value_to_send = {"action": "accept"}          # or
              value_to_send = {"action": "reject", "feedback": "too noisy"}

          elif kind == "done":
              # payload = str path to the final output directory
              pass

    The interrupt() / resume cycle
    ───────────────────────────────
    When the graph reaches human_review_node it calls LangGraph's interrupt(),
    which suspends the graph and checkpoints state to MemorySaver.
    HITLRunner.run() catches the resulting event, yields ("review", payload)
    to the caller, and resumes the graph with Command(resume=decision) once
    the caller sends the human decision back via gen.send().

    This means the graph can be paused indefinitely (waiting for user input)
    without holding any thread or blocking any event loop.
    """

    def __init__(
        self,
        api_key: str,
        sample_size: int = 5,
        output_root: Path = Path("outputs"),
    ) -> None:
        self._api_key = api_key
        self._sample_size = sample_size
        self._output_root = Path(output_root)
        memory = MemorySaver()
        self._graph = build_graph().compile(checkpointer=memory)
        self._config = {"configurable": {"thread_id": f"biovision-{id(self)}"}}

    def run(self, input_folder: str, metadata_content: str) -> Generator:
        """
        Generator that drives the full HITL cycle.

        Yields
        ------
        ("log",    str)   — status line for the UI log pane
        ("review", dict)  — mini-batch results; caller must .send() back
                            {"action": "accept"|"reject", "feedback": str}
        ("done",   str)   — final output directory path

        The caller advances the generator with next() and, after receiving a
        ("review", …) event, resumes it with gen.send(decision_dict).
        """
        initial_state: HITLState = {
            "input_folder":        input_folder,
            "metadata_content":    metadata_content,
            "api_key":             self._api_key,
            "output_root":         str(self._output_root),
            "sample_size":         self._sample_size,
            "messages":            [],
            "technique_name":      "",
            "technique_description": "",
            "technique_code":      "",
            "mini_batch_paths":    [],
            "mini_batch_results":  [],
            "human_decision":      None,
            "human_feedback":      None,
            "rejection_count":     0,
            "code_retry_count":    0,
            "last_error":          None,
            "final_output_dir":    None,
            "status_log":          [],
        }

        pending = initial_state

        while True:
            hit_interrupt = False
            interrupt_value: dict = {}
            final_output_dir: Optional[str] = None

            # Stream graph updates one node at a time
            for event in self._graph.stream(pending, self._config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        # Graph suspended — collect the interrupt payload
                        hit_interrupt = True
                        interrupt_value = node_output[0].value if node_output else {}
                    else:
                        # Forward per-node log messages to the UI
                        for msg in node_output.get("status_log", []):
                            if msg:
                                yield ("log", str(msg))
                        # Track output directory for final completion signal
                        if node_output.get("final_output_dir"):
                            final_output_dir = node_output["final_output_dir"]

            if hit_interrupt:
                # ── HITL pause ───────────────────────────────────────────
                # Yield the review payload to the caller.
                # The caller suspends here (inside the napari thread_worker)
                # until the UI calls worker.send(decision).
                decision = yield ("review", interrupt_value)

                if decision is None:
                    return  # caller abandoned (e.g. Stop button)

                # Resume the graph from the checkpoint with the human decision
                pending = Command(resume=decision)
                continue

            # Graph completed normally
            if final_output_dir:
                yield ("done", final_output_dir)
            else:
                yield ("log", "Agent finished.  No output was produced.")
            return
