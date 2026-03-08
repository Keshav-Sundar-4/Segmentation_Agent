#!/usr/bin/env python
"""
Launch BioVision from the project root:

    python launch.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "UI"
SRC_DIR = UI_DIR / "src"

# Ensure all required paths are on sys.path
for p in [str(ROOT), str(SRC_DIR), str(UI_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import napari
from qtpy.QtCore import QTimer
from biovision_napari.ui.main_widget import BioVisionWidget
from biovision_napari.ui.comparison_panel import ComparisonPanel

viewer = napari.Viewer(title="BioVision")

# Main panel (right dock — Samples, Labels, Masks, Bookmarks, Agent)
main_widget = BioVisionWidget(viewer)
viewer.window.add_dock_widget(main_widget, name="BioVision", area="right")

# Wire agent panel signals (agent panel is embedded as a tab in BioVisionWidget)
agent_panel = main_widget._agent_panel
agent_panel.agent_finished.connect(main_widget._browser._refresh)
agent_panel.input_dir_changed.connect(main_widget._on_agent_input_dir_changed)

# Comparison grid (left dock)
comparison = ComparisonPanel(viewer, main_widget._state)
viewer.window.add_dock_widget(comparison, name="Comparison Grid", area="left")


def _preload():
    yaml_path = UI_DIR / "viewer.yaml"
    if yaml_path.exists():
        main_widget._state.load(str(yaml_path))
        main_widget._load_sample("sample_001")


QTimer.singleShot(500, _preload)

napari.run()
