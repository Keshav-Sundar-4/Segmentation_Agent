"""
tools/search.py — Web-search tool selection for the BioVision agent.

get_search_tool() returns the best available tool:
  1. TavilySearchResults  (requires TAVILY_API_KEY)
  2. DuckDuckGoSearchRun  (free fallback)
  3. None                 (agent proceeds without search)
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_search_tool():
    """Return the highest-quality available web-search tool, or None."""
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
