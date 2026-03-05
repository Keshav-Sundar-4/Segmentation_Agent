"""
Pydantic schemas for structured LLM outputs.

Using `.with_structured_output(Model)` on a LangChain chat model forces the
LLM to return a validated instance of that model rather than raw text.  Each
schema maps 1-to-1 to a set of PipelineState fields so the node can simply
unpack the model into a dict and return it.
"""

from pydantic import BaseModel, Field


class PreprocessingPlan(BaseModel):
    """Structured output produced by the Planner node."""

    title: str = Field(
        description=(
            "A short, descriptive name for the preprocessing pipeline "
            "(e.g. 'CLAHE + Median Denoising + Otsu Thresholding')."
        )
    )
    steps: list[str] = Field(
        description=(
            "Ordered list of preprocessing steps. Each step is a concise "
            "natural-language instruction including specific parameter values "
            "where relevant (e.g. 'Apply Gaussian blur with sigma=1.5 to "
            "reduce high-frequency noise')."
        )
    )
    rationale: str = Field(
        description=(
            "A brief explanation of why these steps — in this order — will "
            "maximise downstream segmentation accuracy for this dataset."
        )
    )


class GeneratedCode(BaseModel):
    """Structured output produced by the Coder node."""

    code: str = Field(
        description=(
            "A complete, self-contained Python script. "
            "MUST read image paths from the environment variable INPUT_DIR "
            "and write processed images to OUTPUT_DIR. "
            "Return ONLY raw Python — no markdown fences, no prose, no "
            "explanatory comments outside the code itself."
        )
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description=(
            "pip package names required at runtime, "
            "e.g. ['scikit-image', 'tifffile', 'numpy']. "
            "Omit packages from the Python standard library."
        ),
    )
