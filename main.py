"""
BioVision — Unconstrained REPL-Driven Image Preprocessing Agent
----------------------------------------------------------------
Reads a user-supplied image folder and metadata file, then spins up a
LangChain ReAct agent that dynamically writes, executes, and debugs
three image-processing pipelines entirely through a Python REPL.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate

from tools import LoggingPythonREPLTool

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("biovision")

# ---------------------------------------------------------------------------
# System / ReAct prompt
# ---------------------------------------------------------------------------
REACT_PROMPT_TEMPLATE = """\
You are an expert image-processing AI engineer specialising in scientific,
medical, and remote-sensing image datasets.

Your mission: analyse the supplied dataset metadata and execute THREE distinct,
optimal preprocessing pipelines through the Python REPL tool.

════════════════════════════════════════════════════════════════
STRICT GUARDRAILS — YOU MUST FOLLOW THESE AT ALL TIMES
════════════════════════════════════════════════════════════════

1. FILESYSTEM JAIL
   • Read images ONLY from the input folder given in the task.
   • Write files ONLY into subfolders of the `outputs/` directory.
   • NEVER delete, move, or modify any file outside `outputs/`.
   • Forbidden calls outside outputs/: os.remove, shutil.rmtree,
     os.unlink, shutil.move, open(..., "w") on arbitrary paths, etc.

2. OUTPUT STRUCTURE
   • Create a descriptive subfolder per pipeline, for example:
       outputs/pipeline1_clahe_otsu/
       outputs/pipeline2_gaussian_denoising/
       outputs/pipeline3_anisotropic_diffusion/
   • Save every processed image into its pipeline subfolder.
   • Use the same filename stem as the source image.

3. CODE QUALITY
   • Each pipeline must be a COMPLETE, self-contained Python script
     with all imports at the top.
   • Do NOT print image arrays, pixel matrices, or large data to the
     console — only print success/failure messages, file names processed,
     and concise metrics (e.g. SSIM, processing time).
   • Use only the pre-installed packages: cv2, skimage, scipy, numpy,
     matplotlib.  Do NOT attempt pip installs.

4. ERROR HANDLING & RETRIES
   • If code fails, read the traceback, fix the root cause, and retry.
   • Maximum THREE attempts per pipeline.
   • After three failures, print "PIPELINE N FAILED: <reason>" and
     continue to the next pipeline.  Do not loop endlessly.

5. PIPELINE DIVERSITY
   • The three pipelines must use meaningfully different algorithmic
     approaches tailored to the domain inferred from the metadata.
   • Example distinct approaches: contrast-limited adaptive histogram
     equalisation (CLAHE) + Otsu thresholding; Gaussian/bilateral
     denoising + adaptive thresholding; anisotropic diffusion +
     morphological operations; colour-space transforms + edge sharpening.

6. EFFICIENCY
   • Process images with a for-loop; do not load all images at once.
   • Prefer in-place numpy operations where possible.

════════════════════════════════════════════════════════════════
WORKFLOW
════════════════════════════════════════════════════════════════

Step 1 — Analyse metadata
  • Read the metadata (provided below in the task) and determine:
    – Image domain / modality (satellite, histology, fluorescence, …)
    – Likely artefacts and challenges
    – Three complementary pipeline strategies

Step 2 — Execute each pipeline
  For pipeline 1, 2, 3:
    a. State the pipeline's purpose and steps.
    b. Write the complete Python script.
    c. Execute it via the REPL.
    d. On error: debug and retry (≤ 3 attempts total).
    e. Report SUCCESS or FAILURE with a one-line reason.

Step 3 — Final summary
  • List which pipelines succeeded, what each does, and where outputs live.

════════════════════════════════════════════════════════════════
TOOLS
════════════════════════════════════════════════════════════════
You have access to the following tools:
{tools}

Use EXACTLY this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (Thought / Action / Action Input / Observation may repeat)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def prompt_path(prompt_text: str, must_exist: bool = True) -> Path:
    """Prompt the user for a path and validate it."""
    while True:
        raw = input(prompt_text).strip()
        if not raw:
            print("Path cannot be empty. Please try again.")
            continue
        path = Path(raw).expanduser().resolve()
        if must_exist and not path.exists():
            print(f"  Path not found: {path}  — please try again.")
            continue
        return path


def read_metadata(metadata_path: Path) -> str:
    """Return the text content of the metadata file."""
    suffix = metadata_path.suffix.lower()
    if suffix == ".pdf":
        raise ValueError(
            "PDF metadata is not yet supported. "
            "Please convert it to a plain-text or Markdown file."
        )
    return metadata_path.read_text(encoding="utf-8")


def build_task(input_folder: Path, metadata_text: str) -> str:
    return (
        f"INPUT FOLDER: {input_folder}\n\n"
        f"DATASET METADATA:\n{metadata_text}\n\n"
        "Analyse the metadata, then execute THREE distinct preprocessing "
        "pipelines. Write and run a complete Python script for each pipeline "
        "via the Python REPL. Save all outputs into descriptive subfolders "
        "inside the `outputs/` directory."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    print("\n── BioVision Preprocessing Agent ──\n")

    input_folder = prompt_path("Enter path to the image folder: ", must_exist=True)
    if not input_folder.is_dir():
        logger.error("Not a directory: %s", input_folder)
        sys.exit(1)

    metadata_path = prompt_path("Enter path to the metadata file: ", must_exist=True)
    if not metadata_path.is_file():
        logger.error("Not a file: %s", metadata_path)
        sys.exit(1)

    logger.info("Image folder : %s", input_folder)
    logger.info("Metadata file: %s", metadata_path)

    metadata_text = read_metadata(metadata_path)
    logger.info("Metadata loaded (%d characters)", len(metadata_text))

    # Ensure outputs/ directory exists so the agent can write into it
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    logger.info("Output root  : %s", outputs_dir.resolve())

    # ── LLM ──────────────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model="claude-opus-4-6",
        max_tokens=8192,
    )

    # ── Tools ────────────────────────────────────────────────────────────
    repl_tool = LoggingPythonREPLTool()
    tools = [repl_tool]

    # ── Agent ────────────────────────────────────────────────────────────
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,          # shows Thought / Action / Observation
        max_iterations=40,     # generous ceiling for 3 pipelines × 3 retries
        handle_parsing_errors=True,
        return_intermediate_steps=False,
    )

    task = build_task(input_folder, metadata_text)
    logger.info("Invoking agent …\n")

    result = executor.invoke({"input": task})

    logger.info("\n── Agent finished ──")
    logger.info("Final answer:\n%s", result.get("output", "<no output>"))


if __name__ == "__main__":
    main()
