"""
BioVision Agent package.

Primary public API
──────────────────
    from core.pipeline import PipelineRunner
    from agents.preprocessing.agent import PreprocessingAgent

    runner = (
        PipelineRunner(api_key="sk-ant-...", sample_size=5)
        .add_agent(PreprocessingAgent())
        .with_hitl()
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
            agent.py    — PreprocessingAgent(BaseAgent)
            nodes.py    — 5 pure node functions
    tools/
        repl.py         — exec_repl() + SafeLoggingREPLTool
        search.py       — get_search_tool()
        sampler.py      — sample_images(), discover_images()
    config.py           — AgentConfig dataclass & multi-provider LLM builder
    main.py             — Legacy CLI ReAct agent (standalone)
"""

from core.pipeline import PipelineRunner
from agents.preprocessing.agent import PreprocessingAgent
from core.state import PipelineState

__all__ = ["PipelineRunner", "PreprocessingAgent", "PipelineState"]
