"""
agents/preprocessing/nodes.py — Pure node functions for the preprocessing agent.

Node inventory
──────────────
  prep_load_node      Discovers mini-batch; truncates metadata.
  prep_research_node  LLM + optional web search → technique selection.
  prep_codegen_node   LLM writes + tests a single-image script (≤3 retries).
  prep_preview_node   Runs script on mini-batch; records before/after paths.
  prep_full_run_node  Batch-executes script on entire dataset; sets final_output.

State fields (all prefixed "prep_")
────────────────────────────────────
  prep_technique_name, prep_technique_description
  prep_technique_code, prep_batch_paths, prep_batch_results
  prep_code_retries,   prep_last_error

Cross-cutting fields used (read-only)
──────────────────────────────────────
  input_folder, metadata_content, api_key, output_root, sample_size,
  messages, hitl_rejection_count, hitl_feedback
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from core.state import IMAGE_EXTENSIONS, MAX_CODE_RETRIES, METADATA_CHAR_LIMIT
from tools.repl import exec_repl
from tools.sampler import sample_images
from tools.search import get_search_tool

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_SMART = "claude-3-7-sonnet-latest"   # code generation, debugging
_MODEL_FAST  = "claude-3-5-haiku-latest"    # technique research, light reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_llm(api_key: str, model: str = _MODEL_SMART) -> ChatAnthropic:
    return ChatAnthropic(model=model, max_tokens=4096, temperature=0.0, api_key=api_key)


def _strip_fence(text: str) -> str:
    """Remove ```python / ``` fences that the LLM sometimes wraps code in."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    return re.sub(r"\n?```$", "", text).strip()


def _has_error(output: str) -> bool:
    lower = output.lower()
    return any(kw in lower for kw in ("traceback", "error:", "exception", "syntaxerror"))


def _safe_dirname(name: str) -> str:
    """Convert a technique name to a safe lowercase directory name."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


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
            "prep_last_error":  str(exc),
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
    llm              = _build_llm(state["api_key"], model=_MODEL_FAST)
    metadata         = state["metadata_content"]
    rejection_count  = state.get("hitl_rejection_count", 0)
    human_feedback   = state.get("hitl_feedback") or ""

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
        "prep_code_retries":          0,
        "prep_last_error":            None,
        "messages": [HumanMessage(content=user_content), AIMessage(content=content)],
        "log": [f"Technique selected: {technique_name}"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — prep_codegen
# ─────────────────────────────────────────────────────────────────────────────

def prep_codegen_node(state) -> dict:
    """
    Code-generation node.

    Reads  : prep_technique_name, prep_technique_description,
             prep_batch_paths, output_root, api_key
    Writes : prep_technique_code, prep_last_error, prep_code_retries, log

    Uses claude-3-7-sonnet-latest for code generation.
    Internal retry loop (up to MAX_CODE_RETRIES attempts).
    """
    if not state.get("prep_batch_paths"):
        return {
            "prep_technique_code": "",
            "prep_last_error":     "No images available for code testing.",
            "log": ["Code generation skipped — no images found."],
        }

    llm               = _build_llm(state["api_key"], model=_MODEL_SMART)
    technique_name    = state["prep_technique_name"]
    technique_desc    = state["prep_technique_description"]
    test_input        = state["prep_batch_paths"][0]

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
                "• Read from INPUT_PATH with cv2.IMREAD_UNCHANGED\n"
                "• Apply the technique to produce a processed numpy array\n"
                "• Save to OUTPUT_PATH (ensure parent dirs exist with os.makedirs)\n"
                "• Print 'SUCCESS: <basename>' after saving\n"
                "• Only use: cv2, skimage, scipy, numpy, matplotlib, os, pathlib\n"
                "• No subprocess, no sys.exit, do not modify INPUT_PATH\n\n"
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
        result = exec_repl(runnable)

        if not _has_error(result):
            logger.info("Code generated successfully on attempt %d.", attempt + 1)
            return {
                "prep_technique_code": code,
                "prep_last_error":     None,
                "prep_code_retries":   attempt + 1,
                "log": [f"Code generated and tested (attempt {attempt + 1})."],
            }

        last_error = result[:800]
        logger.warning(
            "Code attempt %d/%d failed: %s", attempt + 1, MAX_CODE_RETRIES, last_error[:120]
        )

    return {
        "prep_technique_code": code,
        "prep_last_error":     last_error,
        "prep_code_retries":   MAX_CODE_RETRIES,
        "log": [f"Code generation failed after {MAX_CODE_RETRIES} attempts."],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — prep_preview
# ─────────────────────────────────────────────────────────────────────────────

def prep_preview_node(state) -> dict:
    """
    Mini-batch preview node.

    Reads  : prep_technique_code, prep_last_error, prep_batch_paths,
             prep_technique_name, output_root
    Writes : prep_batch_results, log
    """
    if state.get("prep_last_error") or not state.get("prep_technique_code"):
        return {
            "prep_batch_results": [],
            "log": ["Mini-batch skipped — code generation failed."],
        }

    technique_name = state["prep_technique_name"]
    technique_code = state["prep_technique_code"]
    batch_paths    = state["prep_batch_paths"]

    output_dir = Path(state["output_root"]) / f"preview_{_safe_dirname(technique_name)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in batch_paths:
        out_path = output_dir / Path(img_path).name
        runnable = (
            f"INPUT_PATH  = {repr(img_path)}\n"
            f"OUTPUT_PATH = {repr(str(out_path))}\n\n"
            + technique_code
        )
        result  = exec_repl(runnable)
        success = out_path.exists() and not _has_error(result)
        results.append({
            "original_path":  img_path,
            "processed_path": str(out_path) if success else None,
            "success":        success,
        })

    ok = sum(1 for r in results if r["success"])
    return {
        "prep_batch_results": results,
        "log": [
            f"Mini-batch: {ok}/{len(results)} images processed.",
            "Waiting for your review…",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — prep_full_run
# ─────────────────────────────────────────────────────────────────────────────

def prep_full_run_node(state) -> dict:
    """
    Full-dataset execution node.

    Reads  : prep_technique_code, prep_technique_name,
             input_folder, output_root
    Writes : final_output, log
    """
    from datetime import datetime

    technique_name = state["prep_technique_name"]
    technique_code = state["prep_technique_code"]
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
        f"        exec(compile({repr(technique_code)}, '<technique>', 'exec'))\n"
        "        _ok += 1\n"
        "        print(f'[{_i}/{_total}] {_p.name}')\n"
        "    except Exception as _e:\n"
        "        print(f'ERROR [{_i}/{_total}] {_p.name}: {_e}', file=sys.stderr)\n\n"
        "print(f'DONE. {_ok}/{_total} images processed.')\n"
        f"print(f'Output: {output_dir}')\n"
    )

    result   = exec_repl(batch_script)
    log_lines = [ln for ln in result.splitlines() if ln.strip()][:25]

    return {
        "final_output": str(output_dir),
        "log": [
            "Full dataset processing complete.",
            f"Output: {output_dir}",
        ] + log_lines,
    }
