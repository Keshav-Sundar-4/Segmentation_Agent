"""
Custom tool wrappers for the REPL-driven preprocessing agent.
"""

import logging
from langchain_experimental.tools import PythonREPLTool

logger = logging.getLogger(__name__)

REPL_LOG_SEP = "=" * 60


class LoggingPythonREPLTool(PythonREPLTool):
    """PythonREPLTool that logs every code submission and its output at INFO level."""

    name: str = "python_repl"
    description: str = (
        "A Python REPL. Use this to execute Python code. "
        "Input should be a valid Python script. "
        "The output will be whatever is printed to stdout/stderr."
    )

    def _run(self, query: str, **kwargs) -> str:
        logger.info(
            "\n%s\nAGENT → REPL (submitting code):\n%s\n%s",
            REPL_LOG_SEP,
            query,
            REPL_LOG_SEP,
        )
        result = super()._run(query, **kwargs)
        logger.info(
            "\n%s\nREPL → AGENT (output):\n%s\n%s",
            REPL_LOG_SEP,
            result if result.strip() else "<no output>",
            REPL_LOG_SEP,
        )
        return result
