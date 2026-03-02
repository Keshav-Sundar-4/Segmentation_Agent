"""
core/base_agent.py — Abstract base class for all pipeline agents.

HOW TO ADD A NEW AGENT
══════════════════════

1. Create agents/myagent/nodes.py  with pure node functions that read/write
   PipelineState fields prefixed with "myagent_".

2. Create agents/myagent/agent.py:

    from core.base_agent import BaseAgent
    from core.state import PipelineState

    class MyAgent(BaseAgent):
        name         = "myagent"           # unique lowercase identifier
        entry_node   = "myagent_load"      # first node (full setup path)
        retry_node   = "myagent_process"   # re-entry after HITL rejection
        preview_node = "myagent_preview"   # node that produces mini-batch results
        full_run_node= "myagent_full_run"  # node executed after HITL accept
        exit_node    = "myagent_full_run"  # last node (next agent chains here)

        def register(self, graph) -> None:
            from agents.myagent.nodes import (
                myagent_load_node, myagent_process_node,
                myagent_preview_node, myagent_full_run_node,
            )
            from langgraph.graph import START, END
            graph.add_node("myagent_load",     myagent_load_node)
            graph.add_node("myagent_process",  myagent_process_node)
            graph.add_node("myagent_preview",  myagent_preview_node)
            graph.add_node("myagent_full_run", myagent_full_run_node)
            # Wire internal edges — do NOT wire preview_node → full_run_node;
            # PipelineRunner does that (directly or via HumanReviewHook).
            graph.add_edge("myagent_load",    "myagent_process")
            graph.add_edge("myagent_process", "myagent_preview")
            graph.add_edge("myagent_full_run", END)

        def initial_state(self) -> dict:
            return {
                "myagent_some_field": None,
                ...
            }

        @staticmethod
        def review_payload(state) -> dict:
            return {"results": state.get("myagent_preview_results", [])}

3. Add the new "myagent_*" fields to core/state.py PipelineState.

4. Wire the agent into a pipeline:

    runner = (
        PipelineRunner(api_key="…")
        .add_agent(PreprocessingAgent())
        .with_hitl()
        .add_agent(MyAgent())   # ← new agent
        .with_hitl()            # ← optional HITL gate for this agent too
        .build()
    )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from langgraph.graph import StateGraph


class BaseAgent(ABC):
    """
    Contract every pipeline agent must satisfy.

    Class variables
    ───────────────
    name          Unique lowercase identifier (e.g. "preprocessing").
                  Used as a namespace for node names.

    entry_node    Name of the first node in this agent's sub-graph.
                  PipelineRunner wires the previous agent's exit_node
                  (or START for the first agent) into this node.

    retry_node    Node to re-enter after a HITL rejection (skips any
                  one-time setup that entry_node may have done).

    preview_node  Node that produces mini-batch preview results.
                  HumanReviewHook attaches here.  PipelineRunner
                  wires this → full_run_node when no HITL is requested.

    full_run_node Node that processes the complete dataset after Accept.

    exit_node     Last node before the next agent (or END).
                  Usually the same as full_run_node.
    """

    name:          ClassVar[str]
    entry_node:    ClassVar[str]
    retry_node:    ClassVar[str]
    preview_node:  ClassVar[str]
    full_run_node: ClassVar[str]
    exit_node:     ClassVar[str]

    @abstractmethod
    def register(self, graph: StateGraph) -> None:
        """
        Add this agent's nodes and *internal* edges to *graph*.

        Rules
        ─────
        • Do NOT add an edge from preview_node → anything;
          PipelineRunner connects that gap (directly or via HITL hook).
        • Do NOT add an edge from entry_node ← anything;
          PipelineRunner connects the previous agent's exit_node → entry_node.
        • DO add an edge from full_run_node → END (or the next checkpoint).
        """

    def initial_state(self) -> dict:
        """
        Return a dict of default values for all state fields owned by
        this agent.  PipelineRunner merges these into the initial state
        passed to graph.stream().
        """
        return {}

    @staticmethod
    def review_payload(state) -> dict:
        """
        Extract the dict that will be passed to interrupt() when the
        HumanReviewHook pauses the graph at this agent's preview_node.

        The returned dict is yielded to the UI as ("review", payload).
        Override this to expose the fields the UI needs to display.
        """
        return {}
