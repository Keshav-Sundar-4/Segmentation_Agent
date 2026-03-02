"""
BioVision Agent package.

Primary public API
──────────────────
    from core.pipeline import PipelineRunner
    from agents.preprocessing.agent import PreprocessingAgent
    from agents.coding.agent import CodingAgent

    runner = (
        PipelineRunner(api_key="sk-ant-...", sample_size=5)
        .add_agent(PreprocessingAgent())          # load + research only
        .add_agent(CodingAgent())                 # codegen + preview + full run
        .with_hitl(on_reject="prep_research")     # rejection → back to research
        .build()
    )
    gen = runner.run(input_folder="/data/images", metadata_content="...")

    value_to_send = None
    while True:
        try:
            event = next(gen) if value_to_send is None else gen.send(value_to_send)
            value_to_send = None
        except StopIteration:
            break

        kind, payload = event
        if kind == "review":
            value_to_send = {"action": "accept"}
            # or: value_to_send = {"action": "reject", "feedback": "too noisy"}

Package layout
──────────────
    core/
        state.py        — PipelineState TypedDict (namespaced fields)
        base_agent.py   — BaseAgent ABC
        pipeline.py     — PipelineRunner (fluent builder + generator)
        hooks/
            human_review.py  — HumanReviewHook (injectable HITL gate)
    agents/
        preprocessing/
            agent.py    — PreprocessingAgent(BaseAgent)  — research only
            nodes.py    — 2 pure node functions (load, research)
        coding/
            agent.py    — CodingAgent(BaseAgent)
            nodes.py    — 3 pure node functions (generate, preview, full_run)
    tools/
        repl.py         — exec_repl() + SafeLoggingREPLTool
        sandbox.py      — exec_sandboxed() — auto-venv for missing packages
        search.py       — get_search_tool()
        sampler.py      — sample_images(), discover_images()
    config.py           — AgentConfig dataclass & multi-provider LLM builder
    main.py             — Legacy CLI ReAct agent (standalone)
"""

from core.pipeline import PipelineRunner
from agents.preprocessing.agent import PreprocessingAgent
from agents.coding.agent import CodingAgent
from core.state import PipelineState

__all__ = ["PipelineRunner", "PreprocessingAgent", "CodingAgent", "PipelineState"]
