"""
agents/coding/nodes.py — Pure node functions for the CodingAgent.

Node inventory
──────────────
  code_generate_node  LLM writes + tests a single-image script (≤3 retries, sandboxed).
  code_preview_node   Runs script on mini-batch via exec_sandboxed.
  code_full_run_node  Iterates full dataset via exec_sandboxed; sets final_output.

State fields owned (prefixed "code_")
──────────────────────────────────────
  code_script, code_batch_results, code_retries, code_last_error

State fields consumed (read-only)
───────────────────────────────────
  prep_technique_name, prep_technique_description, prep_batch_paths,
  input_folder, output_root, api_key
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from core.state import IMAGE_EXTENSIONS, MAX_CODE_RETRIES
from tools.sandbox import exec_sandboxed

logger = logging.getLogger(__name__)

_MODEL_SMART = "claude-3-7-sonnet-latest"


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(api_key: str) -> ChatAnthropic:
    return ChatAnthropic(model=_MODEL_SMART, max_tokens=4096, temperature=0.0, api_key=api_key)


def _strip_fence(text: str) -> str:
    """Remove ```python / ``` fences that the LLM sometimes wraps code in."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    return re.sub(r"\n?```$", "", text).strip()


def _has_error(output: str) -> bool:
    lower = output.lower()
    return any(kw in lower for kw in ("traceback", "error:", "exception", "syntaxerror"))


def _safe_dirname(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — code_generate
# ─────────────────────────────────────────────────────────────────────────────

def code_generate_node(state) -> dict:
    """
    Code-generation node.

    Reads  : prep_technique_name, prep_technique_description,
             prep_batch_paths, output_root, api_key
    Writes : code_script, code_last_error, code_retries, log

    Uses claude-3-7-sonnet-latest with up to MAX_CODE_RETRIES attempts.
    Each attempt is executed via exec_sandboxed (fast or slow path).
    """
    if not state.get("prep_batch_paths"):
        return {
            "code_script":     "",
            "code_last_error": "No images available for code testing.",
            "log": ["Code generation skipped — no images found."],
        }

    llm            = _build_llm(state["api_key"])
    technique_name = state.get("prep_technique_name", "")
    technique_desc = state.get("prep_technique_description", "")
    test_input     = state["prep_batch_paths"][0]

    output_root = Path(state["output_root"])
    preview_dir = output_root / f"preview_{_safe_dirname(technique_name)}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    test_output = str(preview_dir / Path(test_input).name)

    code       = ""
    last_error = ""

    for attempt in range(MAX_CODE_RETRIES):
        if attempt == 0:
            prompt = (
                f"Write a Python script that applies '{technique_name}' to one image.\n\n"
                f"Technique: {technique_desc}\n\n"
                "The script has these two variables pre-defined — DO NOT reassign them:\n"
                f"    INPUT_PATH  = {repr(test_input)}\n"
                f"    OUTPUT_PATH = {repr(test_output)}\n\n"
                "Requirements:\n"
                "• Read from INPUT_PATH\n"
                "• Apply the technique to produce a processed image\n"
                "• Save to OUTPUT_PATH (ensure parent dirs exist with os.makedirs)\n"
                "• Print 'SUCCESS: <basename>' after saving\n"
                "• No subprocess, no sys.exit, do not modify INPUT_PATH\n\n"
                "You may import any Python package you need (numpy, cv2, skimage, scipy, etc.).\n"
                "Write ONLY the Python code.  No markdown fences.  No explanations."
            )
        else:
            prompt = (
                f"Fix this error in the '{technique_name}' script "
                f"(attempt {attempt + 1}/{MAX_CODE_RETRIES}):\n\n"
                f"Error:\n{last_error}\n\n"
                f"Previous code:\n{code}\n\n"
                "Write ONLY the corrected Python code.  No markdown fences."
            )

        resp = llm.invoke([HumanMessage(content=prompt)])
        code = _strip_fence(
            resp.content if isinstance(resp.content, str) else str(resp.content)
        )

        runnable = (
            f"INPUT_PATH  = {repr(test_input)}\n"
            f"OUTPUT_PATH = {repr(test_output)}\n\n"
            + code
        )
        result = exec_sandboxed(runnable)

        if not _has_error(result):
            logger.info("Code generated successfully on attempt %d.", attempt + 1)
            return {
                "code_script":     code,
                "code_last_error": None,
                "code_retries":    attempt + 1,
                "log": [f"Code generated and tested (attempt {attempt + 1})."],
            }

        last_error = result[:800]
        logger.warning(
            "Code attempt %d/%d failed: %s", attempt + 1, MAX_CODE_RETRIES, last_error[:120]
        )

    return {
        "code_script":     code,
        "code_last_error": last_error,
        "code_retries":    MAX_CODE_RETRIES,
        "log": [f"Code generation failed after {MAX_CODE_RETRIES} attempts."],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — code_preview
# ─────────────────────────────────────────────────────────────────────────────

def code_preview_node(state) -> dict:
    """
    Mini-batch preview node.

    Reads  : code_script, code_last_error, prep_batch_paths,
             prep_technique_name, output_root
    Writes : code_batch_results, log
    """
    if state.get("code_last_error") or not state.get("code_script"):
        return {
            "code_batch_results": [],
            "log": ["Mini-batch skipped — code generation failed."],
        }

    technique_name = state.get("prep_technique_name", "unknown")
    script         = state["code_script"]
    batch_paths    = state.get("prep_batch_paths", [])

    output_dir = Path(state["output_root"]) / f"preview_{_safe_dirname(technique_name)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in batch_paths:
        out_path = output_dir / Path(img_path).name
        runnable = (
            f"INPUT_PATH  = {repr(img_path)}\n"
            f"OUTPUT_PATH = {repr(str(out_path))}\n\n"
            + script
        )
        result  = exec_sandboxed(runnable)
        success = out_path.exists() and not _has_error(result)
        results.append({
            "original_path":  img_path,
            "processed_path": str(out_path) if success else None,
            "success":        success,
        })

    ok = sum(1 for r in results if r["success"])
    return {
        "code_batch_results": results,
        "log": [
            f"Mini-batch: {ok}/{len(results)} images processed.",
            "Waiting for your review…",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — code_full_run
# ─────────────────────────────────────────────────────────────────────────────

def code_full_run_node(state) -> dict:
    """
    Full-dataset execution node.

    Reads  : code_script, prep_technique_name, input_folder, output_root
    Writes : final_output, log
    """
    from datetime import datetime

    technique_name = state.get("prep_technique_name", "unknown")
    script         = state["code_script"]
    input_folder   = Path(state["input_folder"])
    output_root    = Path(state["output_root"])

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"full_{_safe_dirname(technique_name)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_script = (
        "import sys\n"
        "from pathlib import Path\n\n"
        f"_in  = Path({repr(str(input_folder))})\n"
        f"_out = Path({repr(str(output_dir))})\n"
        "_out.mkdir(parents=True, exist_ok=True)\n"
        f"_exts = {repr(IMAGE_EXTENSIONS)}\n"
        "_imgs  = sorted(p for p in _in.rglob('*') if p.is_file() and p.suffix.lower() in _exts)\n"
        "_total = len(_imgs)\n"
        "print(f'Full dataset: {_total} images to process.')\n\n"
        "_ok = 0\n"
        "for _i, _p in enumerate(_imgs, 1):\n"
        "    try:\n"
        "        INPUT_PATH  = str(_p)\n"
        "        OUTPUT_PATH = str(_out / _p.name)\n"
        f"        exec(compile({repr(script)}, '<technique>', 'exec'))\n"
        "        _ok += 1\n"
        "        print(f'[{_i}/{_total}] {_p.name}')\n"
        "    except Exception as _e:\n"
        "        print(f'ERROR [{_i}/{_total}] {_p.name}: {_e}', file=sys.stderr)\n\n"
        "print(f'DONE. {_ok}/{_total} images processed.')\n"
        f"print(f'Output: {output_dir}')\n"
    )

    result    = exec_sandboxed(batch_script)
    log_lines = [ln for ln in result.splitlines() if ln.strip()][:25]

    return {
        "final_output": str(output_dir),
        "log": [
            "Full dataset processing complete.",
            f"Output: {output_dir}",
        ] + log_lines,
    }
