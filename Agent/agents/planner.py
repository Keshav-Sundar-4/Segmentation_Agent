"""
Planner node — reads the raw metadata YAML and outputs a structured
preprocessing plan that maximises segmentation accuracy.

The LLM is bound to PreprocessingPlan via .with_structured_output(), so the
return value is always a validated Pydantic model; no post-processing needed.

The chat model is constructed by llm_factory.make_llm() using the provider /
model settings already in PipelineState.  Planner never decides which LLM to
use — that is deterministic runtime configuration set before the graph runs.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..core.llm_factory import make_llm, resolve_model
from ..core.schema import PreprocessingPlan
from ..core.state import PipelineState

logger = logging.getLogger("biovision.planner")

_SYSTEM_PROMPT = """\
You are an expert bioimage analysis scientist specialising in fluorescence and
brightfield microscopy.

Your task is to design an optimal image preprocessing pipeline given a dataset
metadata description.  The pipeline will be implemented in Python by a
downstream coding agent and applied to every image in the dataset before
segmentation.

Guidelines
----------
- Be specific: include concrete parameter values (kernel sizes, sigma values,
  percentile thresholds, etc.) justified by the metadata.
- Order matters: steps are executed sequentially; account for dependencies
  between steps (e.g. denoising before thresholding).
- Consider the modality (confocal, widefield, phase contrast, H&E, …),
  bit depth, and any artefacts mentioned in the metadata.
- Do NOT write any code — produce only the structured plan.
"""


def planner_node(state: PipelineState) -> dict:
    """LangGraph node: metadata YAML → structured PreprocessingPlan."""
    logger.info("Planner: generating preprocessing plan.")

    provider  = state.get("llm_provider", "anthropic") or "anthropic"
    model     = resolve_model(provider, "planner", state.get("llm_model") or "")
    api_key   = state.get("llm_api_key") or state.get("api_key", "")
    base_url  = state.get("llm_base_url", "")

    llm = make_llm(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        schema=PreprocessingPlan,
    )

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                "Dataset metadata:\n"
                f"```yaml\n{state['metadata_yaml']}\n```\n\n"
                "Generate the optimal preprocessing plan."
            )
        ),
    ]

    try:
        plan: PreprocessingPlan = llm.invoke(messages)
    except Exception as exc:
        logger.error("Planner: LLM call failed — %s", exc)
        return {
            "plan_title": "",
            "plan_steps": [],
            "plan_rationale": "",
            "error": f"Planner error: {exc}",
        }

    logger.info(
        "Planner: plan '%s' generated with %d steps.", plan.title, len(plan.steps)
    )

    return {
        "plan_title": plan.title,
        "plan_steps": plan.steps,
        "plan_rationale": plan.rationale,
        "error": None,
    }
