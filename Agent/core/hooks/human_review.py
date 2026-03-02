"""
core/hooks/human_review.py — Agent-agnostic HITL gate.

HumanReviewHook dynamically injects two nodes and their routing edges
between any pair of existing nodes in a StateGraph.  Neither the hook
nor the agent need to know about each other.

Usage
─────
    hook = HumanReviewHook(
        name="preprocessing",
        max_rejections=3,
        payload_fn=PreprocessingAgent.review_payload,
    )
    hook.inject(
        graph,
        after="prep_preview",
        on_accept="prep_full_run",
        on_reject="prep_research",
    )

The hook adds:
    <after> → hitl_<name>_review
    hitl_<name>_review  → (accept)  → <on_accept>
                        → (reject)  → hitl_<name>_rejected
    hitl_<name>_rejected → (count < max_rejections) → <on_reject>
                         → (count >= max_rejections) → END
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

logger = logging.getLogger(__name__)


class HumanReviewHook:
    """
    Injectable HITL gate.  Completely agent-agnostic.

    Parameters
    ──────────
    name            Short name used to build node names:
                    "preprocessing" → nodes "hitl_preprocessing_review"
                    and "hitl_preprocessing_rejected".
    max_rejections  Graph ends after this many rejections (default 3).
    payload_fn      Callable(state) → dict.  The returned dict is passed
                    to interrupt() and forwarded to the UI as
                    ("review", payload).  Defaults to an empty dict.
    """

    def __init__(
        self,
        name: str,
        max_rejections: int = 3,
        payload_fn: Optional[Callable] = None,
    ) -> None:
        self._name = name
        self._max_rejections = max_rejections
        self._payload_fn: Callable = payload_fn or (lambda _s: {})

    # ── Node name helpers ──────────────────────────────────────────────────

    @property
    def review_node_name(self) -> str:
        return f"hitl_{self._name}_review"

    @property
    def rejected_node_name(self) -> str:
        return f"hitl_{self._name}_rejected"

    # ── Graph injection ────────────────────────────────────────────────────

    def inject(
        self,
        graph: StateGraph,
        *,
        after: str,
        on_accept: str,
        on_reject: str,
    ) -> None:
        """
        Inject the two HITL nodes between *after* and *on_accept*/*on_reject*.

        Call this *before* graph.compile().  The method adds:
            <after> → review_node → on_accept   (if accepted)
                                  → rejected_node → on_reject   (if count < max)
                                                  → END          (if count ≥ max)

        The *after* node must already be registered in *graph* but must
        NOT yet have an outgoing edge to anywhere — that edge is what this
        method provides.
        """
        payload_fn    = self._payload_fn
        max_rejections = self._max_rejections
        review_node   = self.review_node_name
        rejected_node = self.rejected_node_name
        _on_accept    = on_accept
        _on_reject    = on_reject

        # ── Node 1: human_review ──────────────────────────────────────────
        def _review_node(state):
            decision = interrupt(payload_fn(state))
            return {
                "hitl_decision": decision.get("action", "reject"),
                "hitl_feedback": decision.get("feedback", ""),
            }

        # ── Node 2: process_rejection ─────────────────────────────────────
        def _rejected_node(state):
            count    = state.get("hitl_rejection_count", 0) + 1
            feedback = state.get("hitl_feedback") or "none"
            logger.info("HITL rejection #%d.  Feedback: %s", count, feedback)
            return {
                "hitl_rejection_count": count,
                "log": [
                    f"Technique rejected (attempt #{count}).  Feedback: {feedback}.",
                    "Researching a different approach…",
                ],
            }

        # ── Routing functions ─────────────────────────────────────────────
        def _route_after_review(state) -> str:
            return _on_accept if state.get("hitl_decision") == "accept" else rejected_node

        def _route_after_rejection(state) -> str:
            if state.get("hitl_rejection_count", 0) >= max_rejections:
                return END
            return _on_reject

        # ── Wire into graph ───────────────────────────────────────────────
        graph.add_node(review_node,   _review_node)
        graph.add_node(rejected_node, _rejected_node)

        graph.add_edge(after, review_node)
        graph.add_conditional_edges(review_node,   _route_after_review)
        graph.add_conditional_edges(rejected_node, _route_after_rejection)

        logger.debug(
            "HumanReviewHook injected: %s → %s → {%s | %s} → {%s | END}",
            after, review_node, _on_accept, rejected_node, _on_reject,
        )
