"""
agents/coding/agent.py — CodingAgent implementation.

Owns all code generation and sandboxed execution.  PreprocessingAgent hands
it the selected technique via shared state fields:
  prep_technique_name, prep_technique_description, prep_batch_paths

Graph topology (internal edges only)
─────────────────────────────────────
  code_generate → code_preview
  (code_preview → ??? left open — PipelineRunner connects directly or via HITL)
  code_full_run → END

Re-entry after HITL rejection
──────────────────────────────
  HumanReviewHook can route back to any node (default: "code_generate").
  The hitl_worker wires it to "prep_research" so rejection re-researches
  the technique rather than just regenerating code.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from core.base_agent import BaseAgent


class CodingAgent(BaseAgent):
    name          = "coding"
    entry_node    = "code_generate"
    retry_node    = "code_generate"
    preview_node  = "code_preview"
    full_run_node = "code_full_run"
    exit_node     = "code_full_run"

    def register(self, graph: StateGraph) -> None:
        from agents.coding.nodes import (
            code_full_run_node,
            code_generate_node,
            code_preview_node,
        )

        graph.add_node("code_generate", code_generate_node)
        graph.add_node("code_preview",  code_preview_node)
        graph.add_node("code_full_run", code_full_run_node)

        graph.add_edge("code_generate", "code_preview")
        # code_preview → ??? is intentionally left open (PipelineRunner fills it)
        graph.add_edge("code_full_run", END)

    def initial_state(self) -> dict:
        return {
            "code_script":        "",
            "code_batch_results": [],
            "code_retries":       0,
            "code_last_error":    None,
        }

    @staticmethod
    def review_payload(state) -> dict:
        """Payload exposed to the UI at the HITL review gate."""
        return {
            "technique_name":        state.get("prep_technique_name", ""),
            "technique_description": state.get("prep_technique_description", ""),
            "mini_batch_results":    state.get("code_batch_results", []),
        }
