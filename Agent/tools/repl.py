"""
tools/repl.py — Python REPL execution for BioVision agent nodes.

Public surface
──────────────
  exec_repl(code)        Execute code in a fresh REPL; return stdout/stderr.
  SafeLoggingREPLTool    LangChain BaseTool with install-blocking and logging.
"""

from __future__ import annotations

import logging
import re

from langchain_experimental.tools import PythonREPLTool

logger = logging.getLogger(__name__)

_SEP = "─" * 60

# Matches pip/conda install calls anywhere in submitted code
_INSTALL_RE = re.compile(
    r"""
    (
        \bpip\s+install\b           |   # pip install ...
        subprocess\.[a-z_]+\s*\(   |   # subprocess.run/call/Popen(...)
        os\.system\s*\(\s*["']pip  |   # os.system("pip ...")
        \bconda\s+install\b             # conda install ...
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper — used directly by node functions
# ─────────────────────────────────────────────────────────────────────────────

def exec_repl(code: str) -> str:
    """
    Execute *code* in a fresh PythonREPLTool session and return the
    combined stdout/stderr as a string.
    """
    repl = PythonREPLTool()
    try:
        return repl._run(code) or ""
    except Exception as exc:
        return f"REPL error: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# LangChain tool (used by the legacy ReAct agent in main.py)
# ─────────────────────────────────────────────────────────────────────────────

class SafeLoggingREPLTool(PythonREPLTool):
    """
    PythonREPLTool that:
    • Logs every submission and output at INFO level.
    • Intercepts pip-install attempts; blocks or warns depending on config.
    • Truncates excessively long output to keep token usage in check.
    """

    name: str = "python_repl"
    description: str = (
        "Execute Python code in a persistent REPL session. "
        "Input must be a complete, valid Python script. "
        "Stdout/stderr is returned as the observation. "
        "Available packages: cv2, skimage, scipy, numpy, matplotlib, os, sys, pathlib. "
        "Do NOT pip install unless absolutely necessary."
    )

    _allow_installs: bool  = True
    _max_output_chars: int = 4000

    def _run(self, query: str, **kwargs) -> str:
        match = _INSTALL_RE.search(query)
        if match:
            if not self._allow_installs:
                msg = (
                    f"BLOCKED: package-installation call detected ('{match.group().strip()}'). "
                    "Rewrite the script using only pre-installed packages: "
                    "cv2, skimage, scipy, numpy, matplotlib."
                )
                logger.warning("SafeREPL blocked install: %s", match.group().strip())
                return msg
            logger.warning(
                "SafeREPL NOTICE: pip/conda install detected — running anyway. Match: %s",
                match.group().strip(),
            )

        logger.info("\n%s\nAGENT → REPL:\n%s\n%s", _SEP, query, _SEP)
        result = super()._run(query, **kwargs)

        if len(result) > self._max_output_chars:
            result = (
                result[: self._max_output_chars]
                + f"\n[...output truncated at {self._max_output_chars} chars]"
            )

        logger.info("\n%s\nREPL → AGENT:\n%s\n%s", _SEP, result or "<no output>", _SEP)
        return result
