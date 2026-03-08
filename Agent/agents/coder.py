"""
Coder node — reads the structured plan and writes a complete Python script
that implements it.

Primary path: .with_structured_output(GeneratedCode) via the LLM.
Fallback path: plain text generation + regex extraction, used automatically
when structured output fails (common with smaller Ollama models where JSON
encoding of Python code breaks on unescaped newlines / quotes).
"""

from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm_factory import make_llm, resolve_model
from ..core.schema import GeneratedCode
from ..core.state import PipelineState

logger = logging.getLogger("biovision.coder")

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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

# Fallback prompt: avoids structured output — uses a text delimiter format
# that can be reliably parsed with regex even if the code contains quotes,
# newlines, or other JSON-special characters.
_FALLBACK_SYSTEM_PROMPT = """\
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
5. The script must be runnable as-is with `python script.py`.

Output format — follow EXACTLY, no other text before or after:
DEPENDENCIES: numpy, scikit-image, tifffile
CODE:
import os
# ... rest of the script
"""

# ---------------------------------------------------------------------------
# Fallback parser
# ---------------------------------------------------------------------------

_DEPS_RE = re.compile(r"^DEPENDENCIES\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_CODE_RE = re.compile(r"^CODE\s*:\s*\n(.*)", re.DOTALL | re.MULTILINE)


def _parse_fallback_response(text: str) -> GeneratedCode:
    """Extract code and dependencies from the fallback plain-text response."""
    deps: list[str] = []
    dm = _DEPS_RE.search(text)
    if dm:
        raw = dm.group(1).strip()
        if raw.upper() not in ("NONE", "N/A", ""):
            deps = [d.strip() for d in raw.split(",") if d.strip()]

    code = ""
    cm = _CODE_RE.search(text)
    if cm:
        code = cm.group(1).strip()
        # Strip trailing ``` if the model added a code fence
        code = re.sub(r"\n?```\s*$", "", code).strip()
    else:
        # Last resort: look for a fenced code block
        fence = re.search(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL)
        if fence:
            code = fence.group(1).strip()
        else:
            # Use everything after the DEPENDENCIES line
            code = _DEPS_RE.sub("", text).strip()

    return GeneratedCode(code=code, dependencies=deps)


def _coder_fallback(
    state: PipelineState,
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
) -> GeneratedCode:
    """Generate code without structured output and parse the result manually."""
    llm = make_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        schema=None,
    )

    numbered_steps = "\n".join(
        f"  {i + 1}. {step}" for i, step in enumerate(state["plan_steps"])
    )
    error_block = ""
    if state.get("error"):
        error_block = (
            f"\n\nThe previous attempt failed with this error — fix it:\n"
            f"{state['error']}"
        )

    messages = [
        SystemMessage(content=_FALLBACK_SYSTEM_PROMPT),
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

    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    return _parse_fallback_response(text)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


def coder_node(state: PipelineState) -> dict:
    """LangGraph node: PreprocessingPlan fields → validated Python script."""
    logger.info("Coder: generating code for plan '%s'.", state.get("plan_title", ""))

    provider = state.get("llm_provider", "anthropic") or "anthropic"
    model    = resolve_model(provider, "coder", state.get("llm_model") or "")
    api_key  = state.get("llm_api_key") or state.get("api_key", "")
    base_url = state.get("llm_base_url", "")

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

    # Surface the previous sandbox error so the LLM can self-correct on retries.
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

    result: GeneratedCode | None = None
    try:
        result = llm.invoke(messages)
        if not result.code or not result.code.strip():
            raise ValueError("Structured output returned an empty code field.")
    except Exception as exc:
        logger.warning(
            "Coder: structured output failed (%s); switching to text fallback.", exc
        )
        try:
            result = _coder_fallback(state, provider, model, api_key, base_url)
        except Exception as exc2:
            logger.error("Coder: text fallback also failed — %s", exc2)
            return {
                "generated_code": "",
                "code_dependencies": [],
                "error": f"Coder error: {exc2}",
            }

    if not result or not result.code.strip():
        return {
            "generated_code": "",
            "code_dependencies": [],
            "error": "Coder produced an empty script. Try again or switch to a larger model.",
        }

    logger.info(
        "Coder: script generated (%d chars, deps=%s).",
        len(result.code),
        result.dependencies,
    )

    return {
        "generated_code": result.code,
        "code_dependencies": result.dependencies,
        # Clear the previous sandbox error on each fresh attempt.
        "error": None,
    }
