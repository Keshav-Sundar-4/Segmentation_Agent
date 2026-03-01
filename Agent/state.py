"""
state.py — Shared State (the "Clipboard") for the BioVision HITL pipeline.

Every LangGraph node reads from and writes to this TypedDict.
Think of it as a structured, typed memo that passes through the entire graph.

Fields marked `Annotated[list, operator.add]` are *append-only*: each node
can add new items and previous items are never lost.  All other fields are
simply overwritten by whichever node last touched them.

Quick reference
───────────────
  Phase 1 – Setup
    input_folder, metadata_content, api_key, output_root, sample_size
    mini_batch_paths    ← filled by load_metadata_node

  Phase 2 – Research
    technique_name, technique_description   ← filled by research_technique_node
    messages                                ← LLM conversation history

  Phase 3 – Code generation
    technique_code      ← filled/updated by generate_code_node
    code_retry_count, last_error

  Phase 4 – Preview
    mini_batch_results  ← filled by execute_mini_batch_node
                          [{original_path, processed_path, success}, ...]

  Phase 5 – HITL gate
    human_decision      ← "accept" | "reject"  (set after interrupt resumes)
    human_feedback      ← optional free-text from the user
    rejection_count     ← incremented by process_rejection_node

  Phase 6 – Full run
    final_output_dir    ← filled by execute_full_dataset_node

  Cross-cutting
    status_log          ← append-only list; each node pushes human-readable
                          progress lines consumed by the napari log widget
"""

from __future__ import annotations

import operator
from typing import Annotated, List, Optional

from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline-wide constants (single source of truth for all nodes and the graph)
# ─────────────────────────────────────────────────────────────────────────────

MAX_CODE_RETRIES:  int = 3   # internal retry loop inside generate_code_node
MAX_REJECTIONS:    int = 3   # user rejection attempts before the graph gives up
METADATA_CHAR_LIMIT: int = 3000  # truncate metadata before sending to the LLM

IMAGE_EXTENSIONS: frozenset = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
)


# ─────────────────────────────────────────────────────────────────────────────
# State schema
# ─────────────────────────────────────────────────────────────────────────────

class HITLState(TypedDict):
    """
    Typed shared state carried through every node of the HITL graph.

    To add a new field for a new node:
      1. Declare it here with its type.
      2. Use `Annotated[list, operator.add]` if the field is accumulative
         (e.g. a running log), or a plain type if it is simply overwritten.
      3. Initialise it to a sensible default in HITLRunner.run()
         (see graph.py → HITLRunner.run → initial_state dict).
    """

    # ── Phase 1: Inputs ───────────────────────────────────────────────────
    input_folder:      str           # absolute path to the image directory
    metadata_content:  str           # YAML text (truncated to METADATA_CHAR_LIMIT)
    api_key:           str           # Anthropic API key — never logged
    output_root:       str           # root directory for all outputs
    sample_size:       int           # number of images in the mini-batch

    # ── Cross-cutting: LLM conversation history ───────────────────────────
    messages: Annotated[list, operator.add]   # accumulates Human+AI messages

    # ── Phase 2: Technique selection ──────────────────────────────────────
    technique_name:        str
    technique_description: str

    # ── Phase 3: Code generation ──────────────────────────────────────────
    technique_code:    str           # self-contained single-image Python script
                                     # uses INPUT_PATH / OUTPUT_PATH as locals
    code_retry_count:  int
    last_error:        Optional[str]

    # ── Phase 4: Mini-batch preview ───────────────────────────────────────
    mini_batch_paths:   List[str]    # fixed-seed sample of image file paths
    mini_batch_results: List[dict]   # [{original_path, processed_path, success}]

    # ── Phase 5: HITL gate ────────────────────────────────────────────────
    human_decision:  Optional[str]   # "accept" | "reject"
    human_feedback:  Optional[str]   # free-text reason (optional)
    rejection_count: int

    # ── Phase 6: Full run ─────────────────────────────────────────────────
    final_output_dir: Optional[str]

    # ── Cross-cutting: progress log ───────────────────────────────────────
    status_log: Annotated[list, operator.add]  # append-only; each node pushes lines
