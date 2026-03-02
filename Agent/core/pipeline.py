"""
core/pipeline.py — Fluent builder that assembles agents into a LangGraph pipeline.

Quick-start
───────────
    from core.pipeline import PipelineRunner
    from agents.preprocessing.agent import PreprocessingAgent

    runner = (
        PipelineRunner(api_key="sk-ant-…", output_root="outputs")
        .add_agent(PreprocessingAgent())   # register agent nodes
        .with_hitl()                       # inject HITL gate for last agent
        .build()                           # compile the graph
    )

    gen = runner.run("/path/to/images", yaml_metadata_text)

    value_to_send = None
    while True:
        try:
            event = next(gen) if value_to_send is None else gen.send(value_to_send)
            value_to_send = None
        except StopIteration:
            break

        kind, payload = event

        if kind == "log":
            print(payload)                                  # status line
        elif kind == "review":
            value_to_send = {"action": "accept"}            # or "reject"
        elif kind == "done":
            print("Output:", payload)                       # output directory
        elif kind == "error":
            print("Error:", payload)

Generator protocol (unchanged from the original HITLRunner)
────────────────────────────────────────────────────────────
    yield ("log",    str)   → status line for the UI log pane
    yield ("review", dict)  → mini-batch payload; caller sends decision back
    yield ("done",   str)   → final output directory path
    yield ("error",  str)   → fatal error (generator stops after this)

    worker.send({"action": "accept"|"reject", "feedback": "…"})
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from core.base_agent import BaseAgent
from core.hooks.human_review import HumanReviewHook
from core.state import PipelineState

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Fluent builder that assembles BaseAgent instances (and optional HITL
    hooks) into a compiled LangGraph StateGraph, then exposes the
    yield/send generator protocol the napari worker already uses.

    Builder methods
    ───────────────
    .add_agent(agent)     Register the agent's nodes; queue it for chaining.
    .with_hitl(**kw)      Inject a HumanReviewHook for the last added agent.
    .build()              Compile the graph.  Must be the last builder call.

    After .build():
    .run(folder, meta)    Returns a generator that drives the full pipeline.
    """

    def __init__(
        self,
        api_key: str,
        sample_size: int = 5,
        output_root: str | Path = "outputs",
    ) -> None:
        self._api_key     = api_key
        self._sample_size = sample_size
        self._output_root = str(output_root)

        self._graph = StateGraph(PipelineState)
        self._agents: List[BaseAgent]           = []
        self._hitl_flags: List[bool]            = []   # parallel list to _agents
        self._compiled_graph                    = None
        self._config: dict                      = {}

    # ── Builder API ────────────────────────────────────────────────────────

    def add_agent(self, agent: BaseAgent) -> "PipelineRunner":
        """Register *agent*'s nodes in the graph and queue it for wiring."""
        agent.register(self._graph)
        self._agents.append(agent)
        self._hitl_flags.append(False)
        return self

    def with_hitl(self, max_rejections: int = 3) -> "PipelineRunner":
        """
        Inject a HumanReviewHook for the most recently added agent.

        The hook is inserted between that agent's preview_node and
        full_run_node.  On rejection, the graph loops back to retry_node.
        """
        if not self._agents:
            raise RuntimeError("Call add_agent() before with_hitl().")
        self._hitl_flags[-1] = True
        self._max_rejections  = max_rejections
        return self

    def build(self) -> "PipelineRunner":
        """
        Chain agents, inject HITL hooks, and compile the graph.

        Wiring rules
        ────────────
        • First agent's entry_node ← START
        • Each subsequent agent's entry_node ← previous agent's exit_node
        • If agent has HITL:
            preview_node → hitl_<name>_review → full_run_node (accept)
                                              → hitl_<name>_rejected → retry_node (reject, count < max)
                                                                      → END (count ≥ max)
        • If agent has no HITL:
            preview_node → full_run_node  (direct edge)
        """
        if not self._agents:
            raise RuntimeError("No agents registered.  Call add_agent() first.")

        # ── Wire agent chain ────────────────────────────────────────────
        prev_exit: Optional[str] = None

        for agent, use_hitl in zip(self._agents, self._hitl_flags):
            # Connect previous exit → this agent's entry (or START)
            if prev_exit is None:
                self._graph.add_edge(START, agent.entry_node)
            else:
                self._graph.add_edge(prev_exit, agent.entry_node)

            # Connect preview → full_run (directly or via HITL)
            if use_hitl:
                hook = HumanReviewHook(
                    name=agent.name,
                    max_rejections=getattr(self, "_max_rejections", 3),
                    payload_fn=agent.review_payload,
                )
                hook.inject(
                    self._graph,
                    after=agent.preview_node,
                    on_accept=agent.full_run_node,
                    on_reject=agent.retry_node,
                )
            else:
                self._graph.add_edge(agent.preview_node, agent.full_run_node)

            prev_exit = agent.exit_node

        # ── Compile ─────────────────────────────────────────────────────
        memory = MemorySaver()
        self._compiled_graph = self._graph.compile(checkpointer=memory)
        self._config = {"configurable": {"thread_id": f"biovision-{id(self)}"}}
        logger.info("PipelineRunner: graph compiled with %d agent(s).", len(self._agents))
        return self

    # ── Generator API ──────────────────────────────────────────────────────

    def run(self, input_folder: str, metadata_content: str) -> Generator:
        """
        Generator that drives the compiled pipeline.

        Yields  (kind, payload)  tuples — see module docstring.
        Receives decision dicts via gen.send() after ("review", …) events.
        """
        if self._compiled_graph is None:
            raise RuntimeError("Call .build() before .run().")

        # Merge pipeline-wide defaults with per-agent defaults
        initial_state: dict = {
            "input_folder":       input_folder,
            "metadata_content":   metadata_content,
            "api_key":            self._api_key,
            "output_root":        self._output_root,
            "sample_size":        self._sample_size,
            "messages":           [],
            "log":                [],
            "final_output":       None,
            "hitl_decision":      None,
            "hitl_feedback":      None,
            "hitl_rejection_count": 0,
        }
        for agent in self._agents:
            initial_state.update(agent.initial_state())

        pending = initial_state

        while True:
            hit_interrupt   = False
            interrupt_value: dict = {}
            final_output: Optional[str] = None

            for event in self._compiled_graph.stream(
                pending, self._config, stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        hit_interrupt   = True
                        interrupt_value = node_output[0].value if node_output else {}
                    else:
                        for msg in node_output.get("log", []):
                            if msg:
                                yield ("log", str(msg))
                        if node_output.get("final_output"):
                            final_output = node_output["final_output"]

            if hit_interrupt:
                decision = yield ("review", interrupt_value)
                if decision is None:
                    return   # caller abandoned (e.g. Stop button)
                pending = Command(resume=decision)
                continue

            if final_output:
                yield ("done", final_output)
            else:
                yield ("log", "Pipeline finished.  No output was produced.")
            return
