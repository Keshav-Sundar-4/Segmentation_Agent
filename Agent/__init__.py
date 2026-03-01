"""
BioVision Agent package.

Primary public API
──────────────────
    from agent import HITLRunner

    runner = HITLRunner(api_key="sk-ant-...", sample_size=5)
    gen    = runner.run(input_folder="/data/images", metadata_content="...")

    value_to_send = None
    while True:
        try:
            event = next(gen) if value_to_send is None else gen.send(value_to_send)
            value_to_send = None
        except StopIteration:
            break

        kind, payload = event
        if kind == "review":
            # Show payload to user, collect their decision, then:
            value_to_send = {"action": "accept"}
            # or: value_to_send = {"action": "reject", "feedback": "too noisy"}

Package layout
──────────────
    state.py    — HITLState TypedDict  ("the shared clipboard")
    nodes.py    — 7 agent/worker functions  ("the workers")
    graph.py    — build_graph() + HITLRunner  ("the manager")
    tools.py    — REPL execution, web search, image inspector
    sampler.py  — image discovery & deterministic mini-batch sampling
    config.py   — AgentConfig dataclass & multi-provider LLM builder

Note on import paths
────────────────────
The napari worker (UI/workers/hitl_worker.py) adds *this directory* (Agent/)
to sys.path, so individual modules (state, nodes, graph, …) are importable
directly.  When the UI is updated to add the *repo root* to sys.path instead,
the `from agent import HITLRunner` form below will work transparently.
"""

from graph import HITLRunner, build_graph
from state import HITLState

__all__ = ["HITLRunner", "build_graph", "HITLState"]
