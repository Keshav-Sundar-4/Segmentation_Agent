import napari
from pathlib import Path
from qtpy.QtCore import QTimer
from biovision_napari.ui.main_widget import BioVisionWidget
from biovision_napari.ui.comparison_panel import ComparisonPanel
from biovision_napari.ui.llm_chat import LLMChatWidget

viewer = napari.Viewer(title="BioVision")

# Main panel
main_widget = BioVisionWidget(viewer)
viewer.window.add_dock_widget(main_widget, name="BioVision", area="right")

# LLM chat
llm_widget = LLMChatWidget(main_widget._state)
viewer.window.add_dock_widget(llm_widget, name="LLM Chat", area="bottom")

# Comparison grid
comparison = ComparisonPanel(viewer, main_widget._state)
viewer.window.add_dock_widget(comparison, name="Comparison Grid", area="left")

# Delay sample load until after the Qt event loop is running
def _preload():
    yaml_path = Path(__file__).parent / "viewer.yaml"
    main_widget._state.load(str(yaml_path))
    main_widget._load_sample("sample_001")

QTimer.singleShot(500, _preload)

napari.run()
