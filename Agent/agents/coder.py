"""
Coder node — reads the structured plan and writes a complete Python script
that implements it.

The LLM sees ONLY: the plan (title + steps + rationale), and the two directory
paths.  It never sees the raw metadata or previous conversation history so
that its output stays focused on implementation rather than re-analysis.

.with_structured_output(GeneratedCode) guarantees:
  - `code`         : raw Python string (no markdown fences)
  - `dependencies` : list of pip packages needed at runtime

The chat model is constructed by llm_factory.make_llm() using the provider /
model settings already in PipelineState.  Coder never decides which LLM to
use — that is deterministic runtime configuration set before the graph runs.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm_factory import make_llm, resolve_model
from ..core.schema import GeneratedCode
from ..core.state import PipelineState

logger = logging.getLogger("biovision.coder")

_SYSTEM_PROMPT = """\
You are a senior Python engineer specialising in bioimage processing pipelines.

Given a preprocessing plan and two directory paths, write a complete,
self-contained Python script that applies the plan to every image in INPUT_DIR
and saves the results to OUTPUT_DIR.

Hard constraints
----------------
1. Read paths ONLY from the environment variables INPUT_DIR and OUTPUT_DIR
   (use os.environ["INPUT_DIR"] / os.environ["OUTPUT_DIR"]).
2. Preserve the original filename; change the extension only if the output
   format genuinely differs from the input.
3. Support common microscopy formats: TIFF (.tif/.tiff), PNG, JPEG, CZI, LIF.
   Use tifffile for TIFF I/O; fall back to PIL/Pillow for PNG/JPEG.
4. Use only these scientific Python libraries: numpy, scikit-image, tifffile,
   Pillow, opencv-python-headless (cv2), scipy.
5. The script must be runnable as-is with `python script.py` — include all
   necessary imports, no placeholders, no TODO comments.
6. Return ONLY raw Python code.  No markdown fences, no prose outside the
   script itself.
"""


def coder_node(state: PipelineState) -> dict:
    """LangGraph node: PreprocessingPlan fields → validated Python script."""
    logger.info("Coder: generating code for plan '%s'.", state.get("plan_title", ""))

    provider  = state.get("llm_provider", "anthropic") or "anthropic"
    model     = resolve_model(provider, "coder", state.get("llm_model") or "")
    api_key   = state.get("llm_api_key") or state.get("api_key", "")
    base_url  = state.get("llm_base_url", "")

    llm = make_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        schema=GeneratedCode,
    )

    numbered_steps = "\n".join(
        f"  {i + 1}. {step}" for i, step in enumerate(state["plan_steps"])
    )

    # Surface the previous error so the LLM can fix it on retries.
    error_block = ""
    if state.get("error"):
        error_block = (
            f"\n\nThe previous attempt failed with this error — fix it:\n"
            f"```\n{state['error']}\n```"
        )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Plan title: {state['plan_title']}\n"
                f"Rationale: {state['plan_rationale']}\n\n"
                f"Steps:\n{numbered_steps}\n\n"
                f"INPUT_DIR will contain the raw images.\n"
                f"OUTPUT_DIR is where processed images must be written."
                f"{error_block}\n\n"
                "Write the Python script."
            )
        ),
    ]

    result: GeneratedCode = llm.invoke(messages)

    logger.info(
        "Coder: script generated (%d chars, deps=%s).",
        len(result.code),
        result.dependencies,
    )

    return {
        "generated_code": result.code,
        "code_dependencies": result.dependencies,
        # Clear the previous error on each fresh attempt.
        "error": None,
    }
