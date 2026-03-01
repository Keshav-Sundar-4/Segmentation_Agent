"""
hitl_agent.py — LangGraph Human-in-the-Loop preprocessing agent.

Graph topology
──────────────
  START
    └─► load_metadata
          └─► research_technique       (LLM + optional web search)
                └─► generate_code      (LLM writes code; tested with REPL; ≤3 internal retries)
                      ├─► execute_mini_batch
                      │       └─► human_review  ◄── interrupt() — pauses here
                      │               ├── accept ──► execute_full_dataset ──► END
                      │               └── reject ──► process_rejection
                      │                                   └─► research_technique  (loop, ≤3×)
                      └─► END  (code generation failed after all retries)
"""

from __future__ import annotations

import logging
import operator
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Generator, List, Optional

from typing_extensions import TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

logger = logging.getLogger(__name__)

_MAX_CODE_RETRIES = 3
_MAX_REJECTIONS = 3
_METADATA_CHAR_LIMIT = 3000
_IMAGE_EXTS: frozenset = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}
)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


class HITLState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────
    input_folder: str
    metadata_content: str
    api_key: str
    output_root: str
    sample_size: int

    # ── LLM message history (accumulated across nodes) ────────
    messages: Annotated[list, operator.add]

    # ── Technique ─────────────────────────────────────────────
    technique_name: str
    technique_description: str
    technique_code: str           # self-contained single-image Python script

    # ── Mini-batch ────────────────────────────────────────────
    mini_batch_paths: List[str]
    mini_batch_results: List[dict]  # [{original_path, processed_path, success}]

    # ── HITL ──────────────────────────────────────────────────
    human_decision: Optional[str]   # "accept" | "reject"
    human_feedback: Optional[str]
    rejection_count: int

    # ── Code generation ───────────────────────────────────────
    code_retry_count: int
    last_error: Optional[str]

    # ── Output ────────────────────────────────────────────────
    final_output_dir: Optional[str]

    # ── Per-node log messages (accumulates across nodes) ──────
    status_log: Annotated[list, operator.add]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm(api_key: str) -> ChatAnthropic:
    return ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        temperature=0.0,
        api_key=api_key,
    )


def _strip_code_fence(text: str) -> str:
    """Remove ```python / ``` fences if present."""
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    return re.sub(r"\n?```$", "", text).strip()


def _exec_repl(code: str) -> str:
    """Execute code via LangChain's PythonREPLTool and return stdout/stderr."""
    from langchain_experimental.tools import PythonREPLTool  # lazy import

    repl = PythonREPLTool()
    try:
        return repl._run(code) or ""
    except Exception as exc:
        return f"REPL error: {exc}"


def _is_error(output: str) -> bool:
    """Return True if the REPL output contains an error signature."""
    lower = output.lower()
    return any(kw in lower for kw in ("traceback", "error:", "exception", "syntaxerror"))


def _safe_name(name: str) -> str:
    """Convert technique name → filesystem-safe lowercase identifier."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def _get_search_tool():
    """Return best available web search tool, or None."""
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults  # noqa: F401
            return TavilySearchResults(max_results=3)
        except ImportError:
            pass
    try:
        from langchain_community.tools import DuckDuckGoSearchRun  # noqa: F401
        return DuckDuckGoSearchRun()
    except ImportError:
        pass
    return None


def _discover_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _sample_images(folder: Path, n: int = 5, seed: int = 42) -> list[Path]:
    images = _discover_images(folder)
    if not images:
        raise ValueError(f"No images found in '{folder}'.")
    rng = random.Random(seed)
    return rng.sample(images, min(n, len(images)))


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────


def load_metadata_node(state: HITLState) -> dict:
    """Discover mini-batch images and truncate metadata to token budget."""
    folder = Path(state["input_folder"])
    sample_size = state.get("sample_size", 5)
    metadata = state["metadata_content"][:_METADATA_CHAR_LIMIT]

    try:
        batch = _sample_images(folder, n=sample_size, seed=42)
        batch_strs = [str(p) for p in batch]
    except ValueError as exc:
        return {
            "status_log": [f"ERROR: {exc}"],
            "last_error": str(exc),
            "mini_batch_paths": [],
            "metadata_content": metadata,
        }

    return {
        "metadata_content": metadata,
        "mini_batch_paths": batch_strs,
        "status_log": [
            f"Loaded {len(batch_strs)} sample images from {folder}.",
            f"Metadata: {len(metadata)} chars.",
        ],
    }


def research_technique_node(state: HITLState) -> dict:
    """Ask the LLM (with optional web search) to select the best technique."""
    llm = _build_llm(state["api_key"])
    metadata = state["metadata_content"]
    rejection_count = state.get("rejection_count", 0)
    human_feedback = state.get("human_feedback") or ""

    search_tool = _get_search_tool()
    agent_llm = llm.bind_tools([search_tool]) if search_tool else llm

    if rejection_count == 0:
        user_content = (
            f"Dataset metadata:\n{metadata}\n\n"
            "Based on this metadata, select the single best image preprocessing technique.\n"
            "You may search the web first if you need to research specific modalities.\n\n"
            "Respond EXACTLY in this format (no other text):\n"
            "TECHNIQUE_NAME: <short descriptive name>\n"
            "TECHNIQUE_DESCRIPTION: <2-3 sentences: what it does and why it fits this data>"
        )
    else:
        feedback_clause = f"\nUser feedback on rejection #{rejection_count}: {human_feedback}" if human_feedback else ""
        user_content = (
            f"Dataset metadata:\n{metadata}\n"
            f"This is attempt #{rejection_count + 1}. The previous technique was rejected.{feedback_clause}\n\n"
            "Suggest a MEANINGFULLY DIFFERENT preprocessing technique.\n\n"
            "Respond EXACTLY in this format:\n"
            "TECHNIQUE_NAME: <short descriptive name>\n"
            "TECHNIQUE_DESCRIPTION: <2-3 sentences>"
        )

    messages_in = state.get("messages", []) + [HumanMessage(content=user_content)]
    resp = agent_llm.invoke(messages_in)
    content = resp.content if isinstance(resp.content, str) else str(resp.content)

    # If the LLM made tool calls (web search), resolve them and re-invoke
    if getattr(resp, "tool_calls", None) and search_tool:
        tool_messages = []
        for tc in resp.tool_calls:
            try:
                from langchain_core.messages import ToolMessage
                search_result = search_tool.invoke(tc["args"].get("query", "bioimage preprocessing"))
                tool_messages.append(
                    ToolMessage(content=str(search_result), tool_call_id=tc["id"])
                )
            except Exception as exc:
                logger.warning("Search tool error: %s", exc)
        if tool_messages:
            resp2 = llm.invoke(messages_in + [resp] + tool_messages)
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


def generate_code_node(state: HITLState) -> dict:
    """Generate a single-image processing script with an internal retry loop."""
    llm = _build_llm(state["api_key"])
    technique_name = state["technique_name"]
    technique_description = state["technique_description"]
    mini_batch_paths = state.get("mini_batch_paths", [])

    if not mini_batch_paths:
        return {
            "technique_code": "",
            "last_error": "No images available for code testing.",
            "status_log": ["Code generation skipped — no images found."],
        }

    test_input = mini_batch_paths[0]
    output_root = Path(state["output_root"])
    preview_dir = output_root / f"preview_{_safe_name(technique_name)}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    test_output = str(preview_dir / Path(test_input).name)

    code = ""
    last_error = ""

    for attempt in range(_MAX_CODE_RETRIES):
        if attempt == 0:
            prompt = (
                f"Write a Python script that applies '{technique_name}' to one image file.\n\n"
                f"Technique: {technique_description}\n\n"
                "The script uses these two pre-defined variables (do NOT reassign them):\n"
                f"    INPUT_PATH  = {repr(test_input)}\n"
                f"    OUTPUT_PATH = {repr(test_output)}\n\n"
                "Requirements:\n"
                "• Read the image from INPUT_PATH (use cv2.IMREAD_UNCHANGED)\n"
                "• Apply the technique to produce a processed array\n"
                "• Save the result to OUTPUT_PATH (create parent dirs: os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True))\n"
                "• Print 'SUCCESS: <basename>' after saving\n"
                "• Use only: cv2, skimage, scipy, numpy, matplotlib, os, pathlib\n"
                "• Do NOT use subprocess, shutil.rmtree, or modify INPUT_PATH\n\n"
                "Write ONLY the Python code. No markdown. No preamble."
            )
        else:
            prompt = (
                f"Fix this Python error for the '{technique_name}' script "
                f"(attempt {attempt + 1}/{_MAX_CODE_RETRIES}):\n\n"
                f"Error output:\n{last_error}\n\n"
                f"Previous code:\n{code}\n\n"
                "Write ONLY the corrected Python code. No markdown."
            )

        resp = llm.invoke([HumanMessage(content=prompt)])
        code = _strip_code_fence(
            resp.content if isinstance(resp.content, str) else str(resp.content)
        )

        # Inject variable definitions and test
        runnable = (
            f"INPUT_PATH = {repr(test_input)}\n"
            f"OUTPUT_PATH = {repr(test_output)}\n\n"
            + code
        )
        result = _exec_repl(runnable)

        if not _is_error(result):
            logger.info("Code generated successfully on attempt %d.", attempt + 1)
            return {
                "technique_code": code,
                "last_error": None,
                "code_retry_count": attempt + 1,
                "status_log": [f"Code generated and tested (attempt {attempt + 1})."],
            }

        last_error = result[:800]
        logger.warning("Code attempt %d/%d failed: %s", attempt + 1, _MAX_CODE_RETRIES, last_error[:120])

    return {
        "technique_code": code,
        "last_error": last_error,
        "code_retry_count": _MAX_CODE_RETRIES,
        "status_log": [f"Code generation failed after {_MAX_CODE_RETRIES} attempts."],
    }


def execute_mini_batch_node(state: HITLState) -> dict:
    """Run the generated code on all mini-batch images and collect results."""
    if state.get("last_error") or not state.get("technique_code"):
        return {
            "mini_batch_results": [],
            "status_log": ["Mini-batch skipped — code generation failed."],
        }

    technique_name = state["technique_name"]
    technique_code = state["technique_code"]
    mini_batch_paths = state["mini_batch_paths"]
    output_root = Path(state["output_root"])
    output_dir = output_root / f"preview_{_safe_name(technique_name)}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for img_path in mini_batch_paths:
        out_path = output_dir / Path(img_path).name
        runnable = (
            f"INPUT_PATH = {repr(img_path)}\n"
            f"OUTPUT_PATH = {repr(str(out_path))}\n\n"
            + technique_code
        )
        result = _exec_repl(runnable)
        success = out_path.exists() and not _is_error(result)
        results.append({
            "original_path": img_path,
            "processed_path": str(out_path) if success else None,
            "success": success,
        })

    ok = sum(1 for r in results if r["success"])
    return {
        "mini_batch_results": results,
        "status_log": [
            f"Mini-batch complete: {ok}/{len(results)} images processed successfully.",
            "Waiting for your review…",
        ],
    }


def human_review_node(state: HITLState) -> dict:
    """
    Pause execution and present mini-batch results to the human.

    LangGraph's interrupt() suspends the graph here.  The caller resumes it
    with Command(resume={"action": "accept"|"reject", "feedback": str}).
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


def process_rejection_node(state: HITLState) -> dict:
    """Increment rejection counter and log feedback."""
    count = state.get("rejection_count", 0) + 1
    feedback = state.get("human_feedback") or "none"
    return {
        "rejection_count": count,
        "status_log": [
            f"Technique rejected (#{count}). Feedback: {feedback}.",
            "Researching a different approach…",
        ],
    }


def execute_full_dataset_node(state: HITLState) -> dict:
    """Run the accepted technique on every image in the input folder."""
    technique_name = state["technique_name"]
    technique_code = state["technique_code"]
    input_folder = Path(state["input_folder"])
    output_root = Path(state["output_root"])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"full_{_safe_name(technique_name)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a batch runner that applies the single-image technique to every file.
    # technique_code uses INPUT_PATH / OUTPUT_PATH as local variables.
    batch_script = (
        "import sys\n"
        "from pathlib import Path\n\n"
        f"_in  = Path({repr(str(input_folder))})\n"
        f"_out = Path({repr(str(output_dir))})\n"
        "_out.mkdir(parents=True, exist_ok=True)\n"
        "_exts = {'.jpg','.jpeg','.png','.tif','.tiff','.bmp','.gif','.webp'}\n"
        "_imgs = sorted(p for p in _in.rglob('*') if p.is_file() and p.suffix.lower() in _exts)\n"
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
        f"print(f'Output directory: {output_dir}')\n"
    )

    result = _exec_repl(batch_script)
    log_lines = [ln for ln in result.splitlines() if ln.strip()][:25]

    return {
        "final_output_dir": str(output_dir),
        "status_log": ["Full dataset processing complete.", f"Output: {output_dir}"] + log_lines,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Edge routing
# ─────────────────────────────────────────────────────────────────────────────


def _route_after_generate(state: HITLState) -> str:
    # If code generation completely failed, abort early
    if state.get("last_error") and not state.get("technique_code"):
        return END
    return "execute_mini_batch"


def _route_after_review(state: HITLState) -> str:
    return "execute_full_dataset" if state.get("human_decision") == "accept" else "process_rejection"


def _route_after_rejection(state: HITLState) -> str:
    if state.get("rejection_count", 0) >= _MAX_REJECTIONS:
        return END
    return "research_technique"


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    g = StateGraph(HITLState)

    g.add_node("load_metadata",        load_metadata_node)
    g.add_node("research_technique",   research_technique_node)
    g.add_node("generate_code",        generate_code_node)
    g.add_node("execute_mini_batch",   execute_mini_batch_node)
    g.add_node("human_review",         human_review_node)
    g.add_node("process_rejection",    process_rejection_node)
    g.add_node("execute_full_dataset", execute_full_dataset_node)

    g.add_edge(START,                  "load_metadata")
    g.add_edge("load_metadata",        "research_technique")
    g.add_edge("research_technique",   "generate_code")
    g.add_conditional_edges("generate_code",    _route_after_generate)
    g.add_edge("execute_mini_batch",   "human_review")
    g.add_conditional_edges("human_review",     _route_after_review)
    g.add_conditional_edges("process_rejection", _route_after_rejection)
    g.add_edge("execute_full_dataset", END)

    return g


# ─────────────────────────────────────────────────────────────────────────────
# Runner  (consumed by the napari hitl_worker)
# ─────────────────────────────────────────────────────────────────────────────


class HITLRunner:
    """
    Drives the compiled LangGraph graph and exposes a generator interface that
    the napari hitl_worker can iterate and interact with via .send().

    Yields
    ------
    ("log",    str)   — status message for the UI log pane
    ("review", dict)  — mini-batch results; caller must .send() back
                        {"action": "accept"|"reject", "feedback": str}
    ("done",   str)   — final output directory path
    """

    def __init__(
        self,
        api_key: str,
        sample_size: int = 5,
        output_root: Path = Path("outputs"),
    ) -> None:
        self._api_key = api_key
        self._sample_size = sample_size
        self._output_root = Path(output_root)
        memory = MemorySaver()
        self._graph = build_graph().compile(checkpointer=memory)
        self._config = {"configurable": {"thread_id": f"biovision-{id(self)}"}}

    def run(self, input_folder: str, metadata_content: str) -> Generator:
        """
        Generator that drives the full HITL cycle.

        Usage pattern in the caller::

            gen = runner.run(folder, metadata)
            value_to_send = None
            while True:
                try:
                    event = next(gen) if value_to_send is None else gen.send(value_to_send)
                    value_to_send = None
                except StopIteration:
                    break
                kind, payload = event
                if kind == "review":
                    value_to_send = get_decision_from_ui()
        """
        initial_state: HITLState = {
            "input_folder": input_folder,
            "metadata_content": metadata_content,
            "api_key": self._api_key,
            "output_root": str(self._output_root),
            "sample_size": self._sample_size,
            "messages": [],
            "technique_name": "",
            "technique_description": "",
            "technique_code": "",
            "mini_batch_paths": [],
            "mini_batch_results": [],
            "human_decision": None,
            "human_feedback": None,
            "rejection_count": 0,
            "code_retry_count": 0,
            "last_error": None,
            "final_output_dir": None,
            "status_log": [],
        }

        pending = initial_state

        while True:
            hit_interrupt = False
            interrupt_value: dict = {}
            final_output_dir: Optional[str] = None

            for event in self._graph.stream(pending, self._config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        hit_interrupt = True
                        interrupt_value = node_output[0].value if node_output else {}
                    else:
                        for msg in node_output.get("status_log", []):
                            if msg:
                                yield ("log", str(msg))
                        if node_output.get("final_output_dir"):
                            final_output_dir = node_output["final_output_dir"]

            if hit_interrupt:
                # Pause — yield review payload to caller; receive decision via send()
                decision = yield ("review", interrupt_value)
                if decision is None:
                    return  # caller abandoned (e.g. Stop button)
                pending = Command(resume=decision)
                continue

            # Graph completed normally
            if final_output_dir:
                yield ("done", final_output_dir)
            else:
                yield ("log", "Agent finished. No output was produced.")
            return
