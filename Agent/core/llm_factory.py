"""
LLM factory — constructs planner / coder models based on runtime configuration.

Provider-agnostic: the same factory is called from both planner_node and
coder_node.  Model selection is *deterministic runtime configuration*, not
an LLM planning task — it never enters the graph.

Supported providers
-------------------
- "anthropic"  → langchain_anthropic.ChatAnthropic
- "ollama"     → langchain_ollama.ChatOllama
                 (falls back to langchain_openai with OpenAI-compat API if
                 langchain_ollama is not installed)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("biovision.llm_factory")

# ── Default models per provider / role ───────────────────────────────────────

_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "planner": "claude-3-5-sonnet-latest",
        "coder":   "claude-3-7-sonnet-latest",
    },
    "ollama": {
        "planner": "llama3.2",
        "coder":   "llama3.2",
    },
}


# ── Public API ────────────────────────────────────────────────────────────────


def make_llm(
    provider: str,
    model: str,
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0,
    schema: Optional[Any] = None,
):
    """
    Create a chat model for the given provider and optionally bind a structured
    output schema.

    Parameters
    ----------
    provider:
        ``"anthropic"`` or ``"ollama"``.
    model:
        Model identifier (e.g. ``"claude-3-5-sonnet-latest"`` or ``"llama3.2"``).
    api_key:
        Required for Anthropic; ignored for Ollama.
    base_url:
        Ollama base URL.  Defaults to ``http://localhost:11434``.
        Unused for Anthropic.
    temperature:
        Generation temperature (``0`` = deterministic, recommended for planning
        and code generation).
    schema:
        Pydantic model to bind via ``.with_structured_output()``.  If ``None``,
        returns the bare chat model.

    Returns
    -------
    BaseChatModel | RunnableBinding
        A LangChain chat model, or its structured-output wrapper if *schema*
        is given.
    """
    provider = (provider or "anthropic").lower().strip()

    if provider == "anthropic":
        llm = _make_anthropic(model, api_key, temperature)
    elif provider == "ollama":
        llm = _make_ollama(model, base_url, temperature)
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Supported values: 'anthropic', 'ollama'."
        )

    if schema is not None:
        return llm.with_structured_output(schema)
    return llm


def resolve_model(provider: str, role: str, model_override: Optional[str] = None) -> str:
    """
    Return a sensible default model identifier for *provider* / *role*,
    unless *model_override* is non-empty.

    Parameters
    ----------
    provider:
        ``"anthropic"`` or ``"ollama"``.
    role:
        ``"planner"`` or ``"coder"``.
    model_override:
        If provided and non-empty, returned as-is (the caller wins).
    """
    if model_override:
        return model_override
    defaults = _DEFAULTS.get(provider, {})
    fallback = next(iter(defaults.values()), "")
    return defaults.get(role, fallback)


# ── Private helpers ───────────────────────────────────────────────────────────


def _make_anthropic(model: str, api_key: str, temperature: float):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain_anthropic is not installed. "
            "Run: pip install langchain-anthropic"
        ) from exc

    return ChatAnthropic(
        model=model,
        api_key=api_key,
        temperature=temperature,
    )


def _make_ollama(model: str, base_url: str, temperature: float):
    ollama_url = (base_url or "http://localhost:11434").rstrip("/")

    try:
        from langchain_ollama import ChatOllama
        logger.debug("LLM factory: using langchain_ollama.ChatOllama.")
        return ChatOllama(
            model=model,
            base_url=ollama_url,
            temperature=temperature,
        )
    except ImportError:
        logger.warning(
            "langchain_ollama not installed; falling back to langchain_openai "
            "with Ollama's OpenAI-compatible API endpoint."
        )

    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            base_url=ollama_url + "/v1",
            api_key="ollama",   # Ollama ignores the key; openai SDK requires one
            temperature=temperature,
        )
    except ImportError as exc:
        raise ImportError(
            "Neither langchain_ollama nor langchain_openai is installed. "
            "Install one of them to use the Ollama provider:\n"
            "  pip install langchain-ollama   # preferred\n"
            "  pip install langchain-openai   # fallback"
        ) from exc
