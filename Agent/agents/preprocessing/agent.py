"""
agents/preprocessing/agent.py — PreprocessingAgent implementation (research-only).

Registers 2 nodes: prep_load and prep_research.
Code generation and execution are handled by CodingAgent.

PipelineRunner handles:
  • wiring entry_node ← START (or previous agent's exit_node)
  • wiring exit_node → next agent's entry_node (e.g. CodingAgent.entry_node)
"""

from __future__ import annotations

from langgraph.graph import StateGraph

from core.base_agent import BaseAgent


class PreprocessingAgent(BaseAgent):
    """
    Research-only pipeline agent.

    Graph topology (internal edges only)
    ─────────────────────────────────────
      prep_load → prep_research
      (prep_research open end — PipelineRunner wires it to the next agent)

    No HITL on this agent.  HITL is attached to CodingAgent, and rejections
    are routed back to "prep_research" so a new technique is selected.
    """

    name          = "preprocessing"
    entry_node    = "prep_load"
    retry_node    = "prep_research"
    preview_node  = None   # research-only; no preview/full_run
    full_run_node = None
    exit_node     = "prep_research"

    def register(self, graph: StateGraph) -> None:
        from agents.preprocessing.nodes import (
            prep_load_node,
            prep_research_node,
        )

        graph.add_node("prep_load",     prep_load_node)
        graph.add_node("prep_research", prep_research_node)

        graph.add_edge("prep_load", "prep_research")
        # prep_research → ??? is intentionally left open (PipelineRunner fills it)

    def initial_state(self) -> dict:
        return {
            "prep_technique_name":        "",
            "prep_technique_description": "",
            "prep_batch_paths":           [],
        }

    @staticmethod
    def review_payload(state) -> dict:
        # PreprocessingAgent has no HITL gate; this method is never called.
        return {}
