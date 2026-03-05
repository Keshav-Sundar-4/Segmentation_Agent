"""
Planner node — reads the raw metadata YAML and outputs a structured
preprocessing plan that maximises segmentation accuracy.

The LLM is bound to PreprocessingPlan via .with_structured_output(), so the
return value is always a validated Pydantic model; no post-processing needed.
"""

from __future__ import annotations

import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

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

    llm = ChatAnthropic(
        model="claude-3-5-sonnet-latest",
        api_key=state["api_key"],
        temperature=0,
    ).with_structured_output(PreprocessingPlan)

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

    plan: PreprocessingPlan = llm.invoke(messages)

    logger.info(
        "Planner: plan '%s' generated with %d steps.", plan.title, len(plan.steps)
    )

    return {
        "plan_title": plan.title,
        "plan_steps": plan.steps,
        "plan_rationale": plan.rationale,
    }
