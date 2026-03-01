"""
hitl_agent.py — backward-compatibility shim.

The BioVision agent has been refactored into a modular package:
    state.py   — HITLState TypedDict
    nodes.py   — agent/worker node functions
    graph.py   — build_graph() + HITLRunner

This file re-exports the public API so existing callers (e.g. the napari
hitl_worker.py) continue to work without modification:

    from hitl_agent import HITLRunner   # still works

Update your import to the canonical form when convenient:
    from graph import HITLRunner        # direct
    from agent import HITLRunner        # package-level (requires repo root on sys.path)
"""

from graph import HITLRunner, build_graph  # noqa: F401
from state import HITLState               # noqa: F401

__all__ = ["HITLRunner", "build_graph", "HITLState"]
