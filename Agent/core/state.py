"""
PipelineState — the single source of truth passed between every graph node.

Design rules
------------
* All fields are primitives or plain collections so the MemorySaver
  checkpointer can serialise them without custom codecs.
* Pydantic models (PreprocessingPlan, GeneratedCode) are *not* stored directly;
  their data is unpacked into typed scalar / list fields instead.
* The `messages` field uses LangGraph's `add_messages` reducer so appends
  from different nodes never clobber each other.
* Add a new prefix block (e.g. `review_*`) here whenever you introduce a new
  agent — keeps namespacing explicit and avoids key collisions.
"""

from __future__ import annotations

from typing import Annotated, Optional

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class PipelineState(TypedDict):
    # ── Inputs (populated once by the caller; never mutated by nodes) ─────────
    metadata_yaml: str          # raw YAML string from the Napari UI
    input_dir: str              # absolute path to folder of raw images
    output_dir: str             # absolute path where processed images are saved
    api_key: str                # Anthropic API key (not logged / persisted)

    # ── Planner outputs ───────────────────────────────────────────────────────
    plan_title: str             # short name for the preprocessing pipeline
    plan_steps: list[str]       # ordered list of preprocessing instructions
    plan_rationale: str         # why these steps maximise segmentation accuracy

    # ── Coder outputs ─────────────────────────────────────────────────────────
    generated_code: str         # complete, runnable Python script
    code_dependencies: list[str]  # pip packages required by the script

    # ── Executor outputs ──────────────────────────────────────────────────────
    execution_stdout: str
    execution_stderr: str
    execution_success: bool

    # ── Control / retry tracking ──────────────────────────────────────────────
    error: Optional[str]        # last error message; None when clean
    retries: int                # how many times coder has been re-invoked

    # ── Conversation trace ────────────────────────────────────────────────────
    # add_messages reducer: each node can append without overwriting prior msgs
    messages: Annotated[list, add_messages]
