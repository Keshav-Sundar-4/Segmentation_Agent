"""
agents/preprocessing/agent.py — PreprocessingAgent implementation.

Registers 5 nodes and their internal edges into the pipeline graph.
PipelineRunner handles:
  • wiring the entry_node ← previous agent (or START)
  • wiring preview_node → full_run_node (directly or via HumanReviewHook)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from core.base_agent import BaseAgent


class PreprocessingAgent(BaseAgent):
    """
    Image preprocessing pipeline agent.

    Graph topology (internal edges only)
    ─────────────────────────────────────
      prep_load → prep_research → prep_codegen
                                      ├─► [END]         ← code gen failed completely
                                      └─► prep_preview  ← (open end — PipelineRunner connects)
                                                          to full_run directly or via HITL hook
      prep_full_run → END   (or next agent's entry_node if chained)

    Re-entry after HITL rejection
    ──────────────────────────────
      HumanReviewHook loops back to retry_node = "prep_research",
      skipping the one-time image-load step (prep_load).
    """

    name          = "preprocessing"
    entry_node    = "prep_load"
    retry_node    = "prep_research"
    preview_node  = "prep_preview"
    full_run_node = "prep_full_run"
    exit_node     = "prep_full_run"

    def register(self, graph: StateGraph) -> None:
        from agents.preprocessing.nodes import (
            prep_codegen_node,
            prep_full_run_node,
            prep_load_node,
            prep_preview_node,
            prep_research_node,
        )

        # ── Register nodes ───────────────────────────────────────────────
        graph.add_node("prep_load",     prep_load_node)
        graph.add_node("prep_research", prep_research_node)
        graph.add_node("prep_codegen",  prep_codegen_node)
        graph.add_node("prep_preview",  prep_preview_node)
        graph.add_node("prep_full_run", prep_full_run_node)

        # ── Internal edges ───────────────────────────────────────────────
        graph.add_edge("prep_load",     "prep_research")
        graph.add_edge("prep_research", "prep_codegen")

        # Conditional: skip preview if code gen completely failed
        graph.add_conditional_edges("prep_codegen", _route_after_codegen)

        # prep_preview → ??? is intentionally left open.
        # PipelineRunner fills this gap (directly or via HumanReviewHook).

        # Full-run exits to END (PipelineRunner may override if agents are chained)
        graph.add_edge("prep_full_run", END)

    def initial_state(self) -> dict:
        return {
            "prep_technique_name":        "",
            "prep_technique_description": "",
            "prep_technique_code":        "",
            "prep_batch_paths":           [],
            "prep_batch_results":         [],
            "prep_code_retries":          0,
            "prep_last_error":            None,
        }

    @staticmethod
    def review_payload(state) -> dict:
        """Payload exposed to the UI at the HITL review gate."""
        return {
            "technique_name":        state.get("prep_technique_name", ""),
            "technique_description": state.get("prep_technique_description", ""),
            "mini_batch_results":    state.get("prep_batch_results", []),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Module-level routing helper
# ─────────────────────────────────────────────────────────────────────────────

def _route_after_codegen(state) -> str:
    """Skip preview if code generation failed entirely."""
    from langgraph.graph import END as _END
    if state.get("prep_last_error") and not state.get("prep_technique_code"):
        return _END
    return "prep_preview"
