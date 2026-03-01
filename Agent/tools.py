"""
tools.py — Comprehensive tool suite for the BioVision HITL agent.

Tools
─────
  python_repl      Execute arbitrary Python in a persistent REPL session.
                   Pip-install attempts are intercepted and either logged
                   (allow_pip_installs=True) or hard-blocked (=False).

  web_search       Research image-processing literature.
                   Uses Tavily if TAVILY_API_KEY is set, else DuckDuckGo.

  inspect_images   Return image metadata (size, dtype, channels, intensity
                   range) without running a REPL script — saves tokens.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool

if TYPE_CHECKING:
    from config import AgentConfig

logger = logging.getLogger(__name__)

_SEP = "─" * 60

# Pattern that reliably matches pip/conda install invocations inside code
_INSTALL_RE = re.compile(
    r"""
    (
        \bpip\s+install\b           |   # pip install ...
        subprocess\.[a-z_]+\s*\(   |   # subprocess.run/call/Popen(..."install"...)
        os\.system\s*\(\s*["']pip  |   # os.system("pip ...")
        \bconda\s+install\b             # conda install ...
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Safe logging REPL
# ---------------------------------------------------------------------------

class SafeLoggingREPLTool(PythonREPLTool):
    """
    PythonREPLTool that:
      • Logs every code submission and REPL output at INFO level.
      • Intercepts pip-install attempts: logs a warning and (optionally)
        blocks them, preventing silent environment sprawl.
      • Truncates excessively long output to keep token usage in check.
    """

    name: str = "python_repl"
    description: str = (
        "Execute Python code in a persistent REPL session. "
        "Input must be a complete, valid Python script. "
        "Stdout/stderr is returned as the observation. "
        "Available packages: cv2, skimage, scipy, numpy, matplotlib, os, sys, pathlib. "
        "Do NOT pip install unless absolutely necessary — prefer pre-installed libs."
    )

    # Injected by build_tools(); controls whether installs are blocked.
    _allow_installs: bool = True
    _max_output_chars: int = 4000   # truncate to keep context lean

    def _run(self, query: str, **kwargs) -> str:
        # ── Install detection ────────────────────────────────────────────────
        match = _INSTALL_RE.search(query)
        if match:
            if not self._allow_installs:
                msg = (
                    "BLOCKED: package-installation call detected "
                    f"('{match.group().strip()}'). "
                    "Rewrite the script using only pre-installed packages: "
                    "cv2, skimage, scipy, numpy, matplotlib."
                )
                logger.warning("SafeREPL blocked install: %s", match.group().strip())
                return msg
            logger.warning(
                "SafeREPL NOTICE: pip/conda install detected — "
                "running anyway (allow_pip_installs=True). "
                "Match: %s",
                match.group().strip(),
            )

        logger.info("\n%s\nAGENT → REPL:\n%s\n%s", _SEP, query, _SEP)
        result = super()._run(query, **kwargs)

        # ── Output truncation (token efficiency) ─────────────────────────────
        if len(result) > self._max_output_chars:
            truncated = result[: self._max_output_chars]
            result = truncated + f"\n[...output truncated at {self._max_output_chars} chars]"

        logger.info("\n%s\nREPL → AGENT:\n%s\n%s", _SEP, result or "<no output>", _SEP)
        return result


# ---------------------------------------------------------------------------
# Image inspector (no REPL needed → saves tokens)
# ---------------------------------------------------------------------------

class ImageInspectorTool(BaseTool):
    """
    Return key image metadata for a list of file paths without writing
    a REPL script.  Much cheaper than spawning Python to do the same thing.

    Input:  newline-separated absolute file paths (max 10 inspected).
    Output: one line per image: name, WxH, channels, dtype, intensity range.
    """

    name: str = "inspect_images"
    description: str = (
        "Get image metadata (dimensions, channels, dtype, intensity range) "
        "for a list of file paths. Much faster than writing a REPL script. "
        "Input: one absolute file path per line (max 10)."
    )

    def _run(self, query: str, **kwargs) -> str:
        paths = [p.strip() for p in query.splitlines() if p.strip()][:10]
        if not paths:
            return "No paths provided."

        lines: list[str] = []
        for path_str in paths:
            p = Path(path_str)
            if not p.is_file():
                lines.append(f"{p.name}: not found")
                continue
            info = _inspect_single(p)
            lines.append(info)

        return "\n".join(lines)

    async def _arun(self, query: str, **kwargs) -> str:  # type: ignore[override]
        return self._run(query, **kwargs)


def _inspect_single(p: Path) -> str:
    """Return a one-line metadata string for one image file."""
    try:
        import cv2
        import numpy as np  # noqa: F401  (imported for dtype access)

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("cv2 returned None — possibly unsupported format")
        h, w = img.shape[:2]
        c = img.shape[2] if img.ndim == 3 else 1
        mn, mx = int(img.min()), int(img.max())
        size_kb = round(p.stat().st_size / 1024, 1)
        return f"{p.name}: {w}×{h}px  {c}ch  {img.dtype}  range=[{mn},{mx}]  {size_kb}KB"
    except Exception as cv2_err:
        pass  # fall through to PIL

    try:
        from PIL import Image

        with Image.open(str(p)) as img:
            w, h = img.size
            mode = img.mode
            size_kb = round(p.stat().st_size / 1024, 1)
            return f"{p.name}: {w}×{h}px  mode={mode}  {size_kb}KB"
    except Exception as pil_err:
        return f"{p.name}: could not inspect — {pil_err}"


# ---------------------------------------------------------------------------
# Search tool (auto-selects best available)
# ---------------------------------------------------------------------------

def get_search_tool():
    """
    Return the highest-quality available web-search tool, or None.

    Priority:
      1. TavilySearchResults  (TAVILY_API_KEY set in .env)
      2. DuckDuckGoSearchRun  (free, no key)
      3. None                 (agent proceeds without search)
    """
    if os.getenv("TAVILY_API_KEY"):
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
            logger.info("Search tool: Tavily (API key found)")
            return TavilySearchResults(max_results=4)
        except ImportError:
            logger.warning("TAVILY_API_KEY set but langchain-community not installed.")

    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        logger.info("Search tool: DuckDuckGo (free, no key required)")
        return DuckDuckGoSearchRun()
    except ImportError:
        pass

    logger.warning(
        "No search tool available. "
        "Run: pip install duckduckgo-search   OR set TAVILY_API_KEY in .env"
    )
    return None


# ---------------------------------------------------------------------------
# Tool builder (single entry-point used by agent.py)
# ---------------------------------------------------------------------------

def build_tools(cfg: "AgentConfig") -> list:
    """Return the complete ordered tool list for the agent."""
    repl = SafeLoggingREPLTool()
    repl._allow_installs = cfg.allow_pip_installs

    tools = [repl, ImageInspectorTool()]

    search = get_search_tool()
    if search:
        tools.append(search)
        logger.info("Tool loaded: %s", search.name)

    logger.info(
        "Tools loaded: %s",
        ", ".join(t.name for t in tools),
    )
    return tools
