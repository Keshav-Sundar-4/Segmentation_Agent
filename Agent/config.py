"""
config.py — BioVision central configuration.

Every tuneable parameter lives here (or in .env).
Swap the LLM provider, model, or any limit without touching other files.

Supported LLM providers
────────────────────────
  anthropic  →  ANTHROPIC_API_KEY   (default; best quality)
  groq       →  GROQ_API_KEY        (very fast; generous free tier)
  openai     →  OPENAI_API_KEY
  ollama     →  no key required     (local models via Ollama)

Supported search tools (auto-selected)
───────────────────────────────────────
  Tavily      →  TAVILY_API_KEY     (best results)
  DuckDuckGo  →  no key required   (free fallback)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Provider → sensible default model (cheap + fast first)
# ---------------------------------------------------------------------------
_PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-haiku-4-5-20251001",   # cheap, fast, very capable
    "groq":      "llama-3.3-70b-versatile",      # free tier, ~330 tok/s
    "openai":    "gpt-4o-mini",
    "ollama":    "llama3.2",                      # local, free
}


@dataclass(frozen=True)
class AgentConfig:
    """Immutable runtime configuration. Build via :func:`load_config`."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    provider:    str            = "anthropic"
    model:       Optional[str]  = None    # None → use _PROVIDER_DEFAULTS
    max_tokens:  int            = 4096
    temperature: float          = 0.0     # deterministic for code generation

    # ── Sampling ─────────────────────────────────────────────────────────────
    sample_size: int            = 5       # images shown to user for approval
    sample_seed: int            = 42      # fixed → same subset across retries

    # ── Agent limits ─────────────────────────────────────────────────────────
    max_agent_iterations:      int = 20   # ReAct loop steps per technique turn
    max_retries_per_technique: int = 3    # REPL fix attempts before abandoning

    # ── Token efficiency ─────────────────────────────────────────────────────
    metadata_char_limit: int   = 3000     # truncate metadata beyond this

    # ── Output ───────────────────────────────────────────────────────────────
    output_root: Path          = field(default_factory=lambda: Path("outputs"))

    # ── Package installation policy ──────────────────────────────────────────
    allow_pip_installs: bool   = True     # log pip installs but allow them
                                          # set False to hard-block in SafeREPL

    def resolved_model(self) -> str:
        """Return the model ID that will actually be used."""
        return self.model or _PROVIDER_DEFAULTS.get(self.provider, "")

    def build_llm(self):
        """Instantiate and return the LangChain chat model for this config."""
        m   = self.resolved_model()
        kw  = dict(temperature=self.temperature, max_tokens=self.max_tokens)
        p   = self.provider

        if p == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=m, **kw)

        if p == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(model=m, **kw)

        if p == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=m, **kw)

        if p == "ollama":
            from langchain_ollama import ChatOllama
            # Ollama doesn't support max_tokens the same way
            return ChatOllama(model=m, temperature=self.temperature)

        raise ValueError(
            f"Unknown LLM provider: '{p}'. "
            "Set LLM_PROVIDER to one of: anthropic | groq | openai | ollama"
        )

    def describe(self) -> str:
        """Human-readable summary for startup logging."""
        return (
            f"provider={self.provider}  model={self.resolved_model()}  "
            f"sample_size={self.sample_size}  max_iterations={self.max_agent_iterations}"
        )


def load_config() -> AgentConfig:
    """Build :class:`AgentConfig` from environment variables / .env file."""
    return AgentConfig(
        provider             = os.getenv("LLM_PROVIDER",      "anthropic"),
        model                = os.getenv("LLM_MODEL")  or None,
        max_tokens           = int(os.getenv("LLM_MAX_TOKENS",  "4096")),
        temperature          = float(os.getenv("LLM_TEMPERATURE", "0.0")),
        sample_size          = int(os.getenv("SAMPLE_SIZE",     "5")),
        sample_seed          = int(os.getenv("SAMPLE_SEED",     "42")),
        max_agent_iterations = int(os.getenv("MAX_AGENT_ITERATIONS",      "20")),
        max_retries_per_technique = int(os.getenv("MAX_RETRIES_PER_TECHNIQUE", "3")),
        metadata_char_limit  = int(os.getenv("METADATA_CHAR_LIMIT",  "3000")),
        output_root          = Path(os.getenv("OUTPUT_ROOT", "outputs")),
        allow_pip_installs   = os.getenv("ALLOW_PIP_INSTALLS", "true").lower() == "true",
    )
