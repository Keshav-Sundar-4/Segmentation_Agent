"""
nodes.py — Agent Workers for the BioVision HITL pipeline.

Each function in this file IS an agent/worker.  It receives the full
HITLState, does exactly one job, and returns a *partial* dict of the fields
it changed.  LangGraph merges that dict back into the shared state.

Node inventory
──────────────
  load_metadata_node          Discovers the mini-batch; truncates metadata.
  research_technique_node     LLM + optional web search → technique choice.
  generate_code_node          LLM writes + tests a single-image script (≤3 retries).
  execute_mini_batch_node     Runs the script on the mini-batch; records results.
  human_review_node           Pauses execution with interrupt(); waits for the UI.
  process_rejection_node      Logs rejection feedback; increments counter.
  execute_full_dataset_node   Runs the accepted script on every image in the folder.

════════════════════════════════════════════════════════════════════════════════
HOW TO ADD A NEW AGENT NODE
════════════════════════════════════════════════════════════════════════════════

Step 1 — Add any new STATE FIELDS it needs to state.py (HITLState).
         Example: add  qc_report: Optional[str] = None

Step 2 — WRITE the node function here, following the same pattern:

    def quality_control_node(state: HITLState) -> dict:
        \"\"\"
        QC Agent: reads processed images, scores them, flags failures.
        Reads  : mini_batch_results, technique_name, api_key
        Writes : qc_report, status_log
        \"\"\"
        llm = _build_llm(state["api_key"], model=_MODEL_FAST)
        results = state["mini_batch_results"]

        # ... do your work ...
        report = "PASS: all images look good"

        return {
            "qc_report": report,
            "status_log": [f"QC complete: {report}"],
        }

Step 3 — REGISTER the node and WIRE its edges in graph.py:

    g.add_node("quality_control", quality_control_node)
    g.add_edge("execute_mini_batch", "quality_control")   # runs after preview
    g.add_edge("quality_control",    "human_review")       # then waits for user

That is all.  The node automatically receives and returns from the shared
HITLState — no other plumbing is needed.
════════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from state import HITLState, IMAGE_EXTENSIONS, MAX_CODE_RETRIES, METADATA_CHAR_LIMIT
from tools import exec_repl, get_search_tool
from sampler import sample_images

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model selection
#
# Use _MODEL_SMART for tasks that require careful reasoning or code generation.
# Use _MODEL_FAST  for simpler classification / routing tasks.
# Both strings are passed directly to ChatAnthropic(model=...).
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_SMART = "claude-3-7-sonnet-latest"   # code generation, debugging
_MODEL_FAST  = "claude-3-5-haiku-latest"    # technique research, light reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers  (private to this module — not exported)
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
# Node 1 — load_metadata
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata_node(state: HITLState) -> dict:
    """
    Setup agent.

    Reads  : input_folder, metadata_content, sample_size
    Writes : metadata_content (truncated), mini_batch_paths, status_log

    Discovers a deterministic mini-batch of images (fixed seed = 42 so the
    same subset is used on every retry — apples-to-apples comparison).
    Truncates the metadata to METADATA_CHAR_LIMIT before passing it forward.
    """
    folder = Path(state["input_folder"])
    metadata = state["metadata_content"][:METADATA_CHAR_LIMIT]

    try:
        batch = sample_images(folder, n=state.get("sample_size", 5), seed=42)
        batch_strs = [str(p) for p in batch]
    except ValueError as exc:
        return {
            "metadata_content": metadata,
            "mini_batch_paths": [],
            "last_error": str(exc),
            "status_log": [f"ERROR loading images: {exc}"],
        }

    return {
        "metadata_content": metadata,
        "mini_batch_paths": batch_strs,
        "status_log": [
            f"Loaded {len(batch_strs)} sample images from {folder}.",
            f"Metadata: {len(metadata)} chars.",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — research_technique
# ─────────────────────────────────────────────────────────────────────────────

def research_technique_node(state: HITLState) -> dict:
    """
    Research agent.

    Reads  : metadata_content, rejection_count, human_feedback, messages, api_key
    Writes : technique_name, technique_description, messages, status_log

    Uses the fast model (claude-3-5-haiku-latest) — technique selection is a
    lightweight reasoning task.  Optionally invokes a web-search tool so the
    LLM can look up domain-specific preprocessing literature before deciding.

    On the first call (rejection_count == 0) it picks the *best* technique.
    On subsequent calls it is forced to pick a *different* technique, informed
    by the user's rejection feedback.
    """
    llm = _build_llm(state["api_key"], model=_MODEL_FAST)
    metadata = state["metadata_content"]
    rejection_count = state.get("rejection_count", 0)
    human_feedback = state.get("human_feedback") or ""

    search_tool = get_search_tool()
    agent_llm = llm.bind_tools([search_tool]) if search_tool else llm

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
    resp = agent_llm.invoke(messages_in)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    # If the LLM issued a search tool call, execute it and re-invoke
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
            resp2 = llm.invoke(messages_in + [resp] + tool_msgs)
            content = resp2.content if isinstance(resp2.content, str) else str(resp2.content)

    # Parse structured response
    technique_name = "Adaptive Preprocessing"
    technique_description = "A general preprocessing pipeline adapted to the dataset."
    for line in content.splitlines():
        if line.startswith("TECHNIQUE_NAME:"):
            technique_name = line.split(":", 1)[1].strip()
        elif line.startswith("TECHNIQUE_DESCRIPTION:"):
            technique_description = line.split(":", 1)[1].strip()

    return {
        "technique_name": technique_name,
        "technique_description": technique_description,
        "code_retry_count": 0,
        "last_error": None,
        "messages": [HumanMessage(content=user_content), AIMessage(content=content)],
        "status_log": [f"Technique selected: {technique_name}"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — generate_code
# ─────────────────────────────────────────────────────────────────────────────

def generate_code_node(state: HITLState) -> dict:
    """
    Code-generation agent.

    Reads  : technique_name, technique_description, mini_batch_paths,
             output_root, api_key
    Writes : technique_code, last_error, code_retry_count, status_log

    Uses the smart model (claude-3-7-sonnet-latest) — Python code generation
    and iterative debugging requires strong reasoning.

    The generated script must honour two pre-defined variables:
        INPUT_PATH   — absolute path to the source image
        OUTPUT_PATH  — absolute path where the processed image should be saved

    An internal retry loop (up to MAX_CODE_RETRIES attempts) catches REPL
    errors and sends the traceback back to the LLM for self-correction.
    """
    if not state.get("mini_batch_paths"):
        return {
            "technique_code": "",
            "last_error": "No images available for code testing.",
            "status_log": ["Code generation skipped — no images found."],
        }

    llm = _build_llm(state["api_key"], model=_MODEL_SMART)
    technique_name = state["technique_name"]
    technique_description = state["technique_description"]
    test_input = state["mini_batch_paths"][0]

    output_root = Path(state["output_root"])
    preview_dir = output_root / f"preview_{_safe_dirname(technique_name)}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    test_output = str(preview_dir / Path(test_input).name)

    code = ""
    last_error = ""

    for attempt in range(MAX_CODE_RETRIES):
        if attempt == 0:
            prompt = (
                f"Write a Python script that applies '{technique_name}' to one image.\n\n"
                f"Technique: {technique_description}\n\n"
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

        # Prepend variable bindings and test-execute via REPL
        runnable = f"INPUT_PATH = {repr(test_input)}\nOUTPUT_PATH = {repr(test_output)}\n\n{code}"
        result = exec_repl(runnable)

        if not _has_error(result):
            logger.info("Code generated successfully on attempt %d.", attempt + 1)
            return {
                "technique_code": code,
                "last_error": None,
                "code_retry_count": attempt + 1,
                "status_log": [f"Code generated and tested (attempt {attempt + 1})."],
            }

        last_error = result[:800]
        logger.warning(
            "Code attempt %d/%d failed: %s", attempt + 1, MAX_CODE_RETRIES, last_error[:120]
        )

    return {
        "technique_code": code,
        "last_error": last_error,
        "code_retry_count": MAX_CODE_RETRIES,
        "status_log": [f"Code generation failed after {MAX_CODE_RETRIES} attempts."],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — execute_mini_batch
# ─────────────────────────────────────────────────────────────────────────────

def execute_mini_batch_node(state: HITLState) -> dict:
    """
    Preview execution agent.

    Reads  : technique_code, last_error, mini_batch_paths,
             technique_name, output_root
    Writes : mini_batch_results, status_log

    Runs the generated script on every image in the mini-batch and records
    both the original and processed file paths so the UI can display them.
    """
    if state.get("last_error") or not state.get("technique_code"):
        return {
            "mini_batch_results": [],
            "status_log": ["Mini-batch skipped — code generation failed."],
        }

    technique_name = state["technique_name"]
    technique_code = state["technique_code"]
    mini_batch_paths = state["mini_batch_paths"]

    output_dir = Path(state["output_root"]) / f"preview_{_safe_dirname(technique_name)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in mini_batch_paths:
        out_path = output_dir / Path(img_path).name
        runnable = (
            f"INPUT_PATH = {repr(img_path)}\n"
            f"OUTPUT_PATH = {repr(str(out_path))}\n\n"
            + technique_code
        )
        result = exec_repl(runnable)
        success = out_path.exists() and not _has_error(result)
        results.append({
            "original_path": img_path,
            "processed_path": str(out_path) if success else None,
            "success": success,
        })

    ok = sum(1 for r in results if r["success"])
    return {
        "mini_batch_results": results,
        "status_log": [
            f"Mini-batch: {ok}/{len(results)} images processed.",
            "Waiting for your review…",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 — human_review  (HITL gate)
# ─────────────────────────────────────────────────────────────────────────────

def human_review_node(state: HITLState) -> dict:
    """
    Human-in-the-Loop gate.

    Reads  : technique_name, technique_description, mini_batch_results
    Writes : human_decision, human_feedback

    Calls LangGraph's interrupt(), which suspends the entire graph at this
    point and serialises the state to the MemorySaver checkpoint.

    The napari worker (hitl_worker.py) catches the resulting GraphInterrupt,
    yields the payload to the Qt main thread as ("review", payload), and then
    blocks until the UI calls worker.send({"action": "accept"|"reject",
    "feedback": "..."}).

    The graph is resumed by the worker calling:
        graph.stream(Command(resume=decision), config)
    """
    decision = interrupt({
        "technique_name": state["technique_name"],
        "technique_description": state["technique_description"],
        "mini_batch_results": state["mini_batch_results"],
    })
    return {
        "human_decision": decision.get("action", "reject"),
        "human_feedback": decision.get("feedback", ""),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 6 — process_rejection
# ─────────────────────────────────────────────────────────────────────────────

def process_rejection_node(state: HITLState) -> dict:
    """
    Rejection-handling agent.

    Reads  : rejection_count, human_feedback
    Writes : rejection_count (incremented), status_log

    Simply increments the counter and logs the feedback so that
    research_technique_node can incorporate it on the next pass.
    """
    count = state.get("rejection_count", 0) + 1
    feedback = state.get("human_feedback") or "none"
    return {
        "rejection_count": count,
        "status_log": [
            f"Technique rejected (attempt #{count}).  Feedback: {feedback}.",
            "Researching a different approach…",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 7 — execute_full_dataset
# ─────────────────────────────────────────────────────────────────────────────

def execute_full_dataset_node(state: HITLState) -> dict:
    """
    Full-run execution agent.

    Reads  : technique_code, technique_name, input_folder, output_root
    Writes : final_output_dir, status_log

    Builds a batch runner that iterates over every image in the input folder,
    executes the technique code for each via exec(), and saves results to a
    timestamped output directory.

    Safety guarantee: the technique code is executed with INPUT_PATH and
    OUTPUT_PATH pointing into output_root — it never receives write access to
    the original input_folder.
    """
    from datetime import datetime

    technique_name = state["technique_name"]
    technique_code = state["technique_code"]
    input_folder = Path(state["input_folder"])
    output_root = Path(state["output_root"])

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    result = exec_repl(batch_script)
    log_lines = [ln for ln in result.splitlines() if ln.strip()][:25]

    return {
        "final_output_dir": str(output_dir),
        "status_log": [
            "Full dataset processing complete.",
            f"Output: {output_dir}",
        ] + log_lines,
    }
