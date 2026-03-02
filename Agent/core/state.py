"""
core/state.py — Shared PipelineState for the BioVision pipeline.

Every LangGraph node reads from and writes to this TypedDict.

Fields use a short prefix to identify which agent owns them:
  (no prefix) — pipeline-wide inputs and cross-cutting fields
  hitl_        — HITL gate state (injected by HumanReviewHook)
  prep_        — PreprocessingAgent fields
  seg_         — (future) SegmentationAgent
  qc_          — (future) QCAgent

To add state for a new agent, append a 4-line prefix block here and
reference the new fields in your agent's nodes and initial_state().

Fields marked Annotated[list, operator.add] are append-only (each node
pushes new items; previous items are never lost). All other fields are
simply overwritten by whichever node last set them.
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional

from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-wide constants (single source of truth)
# ─────────────────────────────────────────────────────────────────────────────

MAX_CODE_RETRIES:    int = 3      # internal retry loop in code_generate_node
MAX_REJECTIONS:      int = 3      # user rejections before the graph gives up
METADATA_CHAR_LIMIT: int = 3000  # truncate metadata before sending to LLM

IMAGE_EXTENSIONS: frozenset = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
)


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    """
    Flat, namespaced shared state for the entire pipeline graph.

    All fields are optional at graph start; each node only touches
    the fields it owns (returning a partial dict).
    """

    # ── Pipeline-wide inputs ──────────────────────────────────────────────
    input_folder:     str            # absolute path to the image directory
    metadata_content: str            # YAML text (truncated before LLM use)
    api_key:          str            # Anthropic API key — never logged
    output_root:      str            # root directory for all outputs
    sample_size:      int            # images in the mini-batch

    # ── Cross-cutting: LLM conversation history ───────────────────────────
    messages: Annotated[list, operator.add]  # accumulates Human+AI messages

    # ── Cross-cutting: progress log ───────────────────────────────────────
    log: Annotated[list, operator.add]       # append-only; UI consumes this

    # ── Cross-cutting: final output ───────────────────────────────────────
    final_output: Optional[str]              # path to final output directory

    # ── HITL gate (written by HumanReviewHook nodes) ─────────────────────
    hitl_decision:        Optional[str]      # "accept" | "reject"
    hitl_feedback:        Optional[str]      # free-text from the user
    hitl_rejection_count: int                # incremented per rejection

    # ── PreprocessingAgent fields (research-only) ─────────────────────────
    prep_technique_name:        str
    prep_technique_description: str
    prep_batch_paths:           List[str]    # fixed-seed mini-batch image paths

    # ── CodingAgent fields ────────────────────────────────────────────────
    code_script:        str                  # single-image script using INPUT_PATH/OUTPUT_PATH
    code_batch_results: List[dict]           # [{original_path, processed_path, success}]
    code_retries:       int                  # attempts used in current codegen loop
    code_last_error:    Optional[str]
