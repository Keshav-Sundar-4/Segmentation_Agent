"""
Comparison panel: up to 4 napari viewers stacked vertically.
Each slot is independently selectable (model source) and can be hidden.
Z/T position and camera are synchronised across all visible slots.
Masks are loaded as read-only label overlays.
"""
from __future__ import annotations

import contextlib
from typing import Optional

import napari
import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from biovision_napari.state.project_state import ProjectState

NUM_SLOTS = 4


# ---------------------------------------------------------------------------
# Camera helpers (napari 0.6.x has no get_state / set_state on Camera)
# ---------------------------------------------------------------------------

def _read_camera(cam) -> dict:
    """Snapshot camera state as plain Python scalars."""
    return {
        "center": tuple(float(v) for v in cam.center),
        "zoom": float(cam.zoom),
        "angles": tuple(float(v) for v in cam.angles),
    }


def _write_camera(cam, state: dict) -> None:
    """Apply a camera state snapshot, ignoring errors from pydantic validation."""
    with contextlib.suppress(Exception):
        cam.center = state["center"]
    with contextlib.suppress(Exception):
        cam.zoom = state["zoom"]
    with contextlib.suppress(Exception):
        cam.angles = state["angles"]


class _SlotWidget(QWidget):
    """One comparison slot: header controls + embedded napari viewer."""

    def __init__(self, slot_idx: int, parent_panel: "ComparisonPanel") -> None:
        super().__init__()
        self._idx = slot_idx
        self._panel = parent_panel
        self._viewer: Optional[napari.Viewer] = None
        self._visible = True

        self._build_ui()
        self._init_viewer()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Header row
        header = QHBoxLayout()
        self._chk_visible = QCheckBox(f"Slot {self._idx + 1}")
        self._chk_visible.setChecked(True)
        self._chk_visible.toggled.connect(self._on_visibility_toggled)
        header.addWidget(self._chk_visible)

        self._combo_model = QComboBox()
        self._combo_model.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._combo_model.addItem("— none —")
        self._combo_model.currentTextChanged.connect(self._on_model_changed)
        header.addWidget(self._combo_model)
        layout.addLayout(header)

        # Viewer container
        self._viewer_container = QWidget()
        self._viewer_container.setMinimumHeight(200)
        self._viewer_layout = QVBoxLayout(self._viewer_container)
        self._viewer_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._viewer_container)

    def _init_viewer(self) -> None:
        self._viewer = napari.Viewer(show=False)
        # Embed the Qt window into our container
        qt_win = self._viewer.window._qt_window
        qt_win.setParent(self._viewer_container)
        qt_win.setWindowFlags(Qt.WindowType.Widget)
        qt_win.show()
        self._viewer_layout.addWidget(qt_win)

    # ------------------------------------------------------------------

    def _on_visibility_toggled(self, checked: bool) -> None:
        self._visible = checked
        self._viewer_container.setVisible(checked)

    def _on_model_changed(self, model_name: str) -> None:
        if model_name == "— none —":
            self._clear_overlays()
        else:
            self._panel._load_model_overlay(self._idx, model_name)

    def _clear_overlays(self) -> None:
        if self._viewer is None:
            return
        labels_layers = [
            l for l in self._viewer.layers if hasattr(l, "selected_label")
        ]
        for l in labels_layers:
            self._viewer.layers.remove(l)

    # ------------------------------------------------------------------
    # Public API

    @property
    def viewer(self) -> Optional[napari.Viewer]:
        return self._viewer

    @property
    def is_visible(self) -> bool:
        return self._visible

    def set_model_options(self, models: list[str]) -> None:
        current = self._combo_model.currentText()
        self._combo_model.blockSignals(True)
        self._combo_model.clear()
        self._combo_model.addItem("— none —")
        for m in models:
            self._combo_model.addItem(m)
        # Restore previous selection if still valid
        idx = self._combo_model.findText(current)
        self._combo_model.setCurrentIndex(max(0, idx))
        self._combo_model.blockSignals(False)

    def load_image(self, arr: np.ndarray, name: str, axis_order: str) -> None:
        if self._viewer is None:
            return
        self._viewer.layers.clear()
        self._viewer.add_image(arr, name=name, colormap="gray")
        object.__setattr__(self._viewer, "_biovision_axis_order", axis_order)

    def load_overlay(self, arr: np.ndarray, name: str) -> None:
        if self._viewer is None:
            return
        existing = [l for l in self._viewer.layers if l.name == name]
        for l in existing:
            self._viewer.layers.remove(l)
        lyr = self._viewer.add_labels(arr, name=name, opacity=0.4)
        lyr.editable = False


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class ComparisonPanel(QWidget):
    """
    Dock widget containing 4 comparison viewer slots stacked vertically.
    Exposes sync logic for Z/T and camera.
    """

    def __init__(
        self,
        primary_viewer: napari.Viewer,
        state: ProjectState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._primary = primary_viewer
        self._state = state
        self._slots: list[_SlotWidget] = []
        self._syncing = False  # re-entrancy guard

        self._build_ui()
        self._connect_primary_sync()
        self._state.config_changed.connect(self._on_config_changed)
        self._state.sample_changed.connect(self._on_sample_changed)

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        btn_refresh = QPushButton("Refresh model list")
        btn_refresh.clicked.connect(self._refresh_model_list)
        layout.addWidget(btn_refresh)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        self._slots_layout = QVBoxLayout(inner)
        self._slots_layout.setSpacing(6)

        for i in range(NUM_SLOTS):
            slot = _SlotWidget(i, self)
            self._slots.append(slot)
            self._slots_layout.addWidget(slot)

        scroll.setWidget(inner)
        layout.addWidget(scroll)

        # Connect slot viewer dims/camera → sync
        for slot in self._slots:
            self._connect_slot_sync(slot)

    # ------------------------------------------------------------------
    # Sync logic
    # ------------------------------------------------------------------

    def _connect_primary_sync(self) -> None:
        """Sync from primary viewer to all comparison slots."""
        self._primary.dims.events.current_step.connect(
            lambda e: self._on_primary_dims_changed(e.value)
        )
        self._primary.camera.events.connect(
            lambda e: self._on_primary_camera_changed()
        )

    def _connect_slot_sync(self, slot: _SlotWidget) -> None:
        """Sync from a comparison slot back to primary and other slots."""
        if slot.viewer is None:
            return

        def on_dims(event):
            if not self._syncing:
                self._propagate_dims(event.value, source_slot=slot)

        def on_camera(event):
            if not self._syncing:
                self._propagate_camera(source_slot=slot)

        slot.viewer.dims.events.current_step.connect(on_dims)
        slot.viewer.camera.events.connect(on_camera)

    def _on_primary_dims_changed(self, step: tuple) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            for slot in self._slots:
                if slot.viewer and slot.is_visible:
                    with contextlib.suppress(Exception):
                        slot.viewer.dims.current_step = step
        finally:
            self._syncing = False

    def _on_primary_camera_changed(self) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            state = _read_camera(self._primary.camera)
            for slot in self._slots:
                if slot.viewer and slot.is_visible:
                    with contextlib.suppress(Exception):
                        _write_camera(slot.viewer.camera, state)
        finally:
            self._syncing = False

    def _propagate_dims(self, step: tuple, source_slot: _SlotWidget) -> None:
        self._syncing = True
        try:
            with contextlib.suppress(Exception):
                self._primary.dims.current_step = step
            for slot in self._slots:
                if slot is not source_slot and slot.viewer and slot.is_visible:
                    with contextlib.suppress(Exception):
                        slot.viewer.dims.current_step = step
        finally:
            self._syncing = False

    def _propagate_camera(self, source_slot: _SlotWidget) -> None:
        if source_slot.viewer is None:
            return
        self._syncing = True
        try:
            cam_state = _read_camera(source_slot.viewer.camera)
            with contextlib.suppress(Exception):
                _write_camera(self._primary.camera, cam_state)
            for slot in self._slots:
                if slot is not source_slot and slot.viewer and slot.is_visible:
                    with contextlib.suppress(Exception):
                        _write_camera(slot.viewer.camera, cam_state)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------
    # Config / sample change
    # ------------------------------------------------------------------

    def _on_config_changed(self, config) -> None:
        self._refresh_model_list()

    def _on_sample_changed(self, sample_id: str) -> None:
        self._refresh_model_list()

    def _refresh_model_list(self) -> None:
        """Discover model output folders for the active sample."""
        cfg = self._state.config
        sample_id = self._state.active_sample
        if cfg is None or sample_id is None:
            return

        from biovision_napari.io.sample_discovery import _discover_models
        from pathlib import Path

        project_root = Path(cfg.paths.dataset_root).parent
        models = _discover_models(sample_id, project_root)

        for slot in self._slots:
            slot.set_model_options(models)

    def _load_model_overlay(self, slot_idx: int, model_name: str) -> None:
        """Load model prediction masks into a slot as read-only overlays."""
        cfg = self._state.config
        sample_id = self._state.active_sample
        if cfg is None or sample_id is None:
            return

        from pathlib import Path
        import tifffile

        model_dir = Path(cfg.paths.output_root) / model_name / sample_id
        slot = self._slots[slot_idx]

        for tif in model_dir.glob("*.tif"):
            try:
                arr = tifffile.imread(str(tif))
                slot.load_overlay(arr, name=f"{model_name}/{tif.stem}")
            except Exception:
                pass
