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
    sample_dir: str             # temp dir holding 1-2 sampled images for sandbox

    # ── LLM runtime configuration ─────────────────────────────────────────────
    # These are set once by the caller and read by planner_node / coder_node.
    # They are NEVER mutated by graph nodes — provider selection is deterministic
    # runtime config, not an LLM planning task.
    llm_provider: str           # "anthropic" | "ollama"
    llm_model: str              # resolved model id; empty → role-specific default
    llm_api_key: str            # Anthropic API key (never logged or persisted)
    llm_base_url: str           # Ollama base URL; empty → "http://localhost:11434"

    # Deprecated: kept for backward compatibility with callers that pass api_key=.
    # The factory reads llm_api_key first and falls back to api_key.
    api_key: str

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
    validated_dependencies: list[str]  # deps confirmed to install/run in sandbox

    # ── Control / retry tracking ──────────────────────────────────────────────
    error: Optional[str]        # last error message; None when clean
    retries: int                # how many times coder has been re-invoked

    # ── Conversation trace ────────────────────────────────────────────────────
    # add_messages reducer: each node can append without overwriting prior msgs
    messages: Annotated[list, add_messages]
