"""
agents/preprocessing/nodes.py — Pure node functions for the preprocessing agent.

Node inventory
──────────────
  prep_load_node      Discovers mini-batch; truncates metadata.
  prep_research_node  LLM + optional web search → technique selection.

State fields (all prefixed "prep_")
────────────────────────────────────
  prep_technique_name, prep_technique_description, prep_batch_paths

Cross-cutting fields used (read-only)
──────────────────────────────────────
  input_folder, metadata_content, api_key, sample_size,
  messages, hitl_rejection_count, hitl_feedback
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from core.state import METADATA_CHAR_LIMIT
from tools.sampler import sample_images
from tools.search import get_search_tool

logger = logging.getLogger(__name__)

_MODEL_FAST = "claude-3-5-haiku-latest"


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(api_key: str) -> ChatAnthropic:
    return ChatAnthropic(model=_MODEL_FAST, max_tokens=4096, temperature=0.0, api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — prep_load
# ─────────────────────────────────────────────────────────────────────────────

def prep_load_node(state) -> dict:
    """
    Setup node.

    Reads  : input_folder, metadata_content, sample_size
    Writes : metadata_content (truncated), prep_batch_paths, log
    """
    folder   = Path(state["input_folder"])
    metadata = state["metadata_content"][:METADATA_CHAR_LIMIT]

    try:
        batch     = sample_images(folder, n=state.get("sample_size", 5), seed=42)
        batch_str = [str(p) for p in batch]
    except ValueError as exc:
        return {
            "metadata_content": metadata,
            "prep_batch_paths": [],
            "log": [f"ERROR loading images: {exc}"],
        }

    return {
        "metadata_content": metadata,
        "prep_batch_paths": batch_str,
        "log": [
            f"Loaded {len(batch_str)} sample images from {folder}.",
            f"Metadata: {len(metadata)} chars.",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — prep_research
# ─────────────────────────────────────────────────────────────────────────────

def prep_research_node(state) -> dict:
    """
    Research node.

    Reads  : metadata_content, hitl_rejection_count, hitl_feedback,
             messages, api_key
    Writes : prep_technique_name, prep_technique_description, messages, log
    """
    llm             = _build_llm(state["api_key"])
    metadata        = state["metadata_content"]
    rejection_count = state.get("hitl_rejection_count", 0)
    human_feedback  = state.get("hitl_feedback") or ""

    search_tool = get_search_tool()
    agent_llm   = llm.bind_tools([search_tool]) if search_tool else llm

    if rejection_count == 0:
        user_content = (
            f"Dataset metadata:\n{metadata}\n\n"
            "Based on this metadata, select the single best image preprocessing technique.\n"
            "You may search the web first if you need to research the specific imaging modality.\n\n"
            "Respond ONLY in this exact format — no other text:\n"
            "TECHNIQUE_NAME: <short descriptive name>\n"
            "TECHNIQUE_DESCRIPTION: <2-3 sentences: what it does and why it fits this data>"
        )
    else:
        feedback_clause = (
            f"\nUser rejection #{rejection_count} feedback: {human_feedback}"
            if human_feedback else ""
        )
        user_content = (
            f"Dataset metadata:\n{metadata}\n"
            f"Attempt #{rejection_count + 1} — the previous technique was rejected.{feedback_clause}\n\n"
            "Suggest a MEANINGFULLY DIFFERENT preprocessing technique.\n\n"
            "Respond ONLY in this exact format:\n"
            "TECHNIQUE_NAME: <short descriptive name>\n"
            "TECHNIQUE_DESCRIPTION: <2-3 sentences>"
        )

    messages_in = state.get("messages", []) + [HumanMessage(content=user_content)]
    resp        = agent_llm.invoke(messages_in)
    content     = resp.content if isinstance(resp.content, str) else str(resp.content)

    # Handle tool calls (web search)
    if getattr(resp, "tool_calls", None) and search_tool:
        from langchain_core.messages import ToolMessage
        tool_msgs = []
        for tc in resp.tool_calls:
            try:
                result = search_tool.invoke(tc["args"].get("query", "bioimage preprocessing"))
                tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
            except Exception as exc:
                logger.warning("Search tool error: %s", exc)
        if tool_msgs:
            resp2   = llm.invoke(messages_in + [resp] + tool_msgs)
            content = resp2.content if isinstance(resp2.content, str) else str(resp2.content)

    # Parse structured response
    technique_name        = "Adaptive Preprocessing"
    technique_description = "A general preprocessing pipeline adapted to the dataset."
    for line in content.splitlines():
        if line.startswith("TECHNIQUE_NAME:"):
            technique_name = line.split(":", 1)[1].strip()
        elif line.startswith("TECHNIQUE_DESCRIPTION:"):
            technique_description = line.split(":", 1)[1].strip()

    return {
        "prep_technique_name":        technique_name,
        "prep_technique_description": technique_description,
        "messages": [HumanMessage(content=user_content), AIMessage(content=content)],
        "log": [f"Technique selected: {technique_name}"],
    }
