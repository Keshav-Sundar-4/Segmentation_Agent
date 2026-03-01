"""
Label layer controls dock widget.
- Lists all label layers defined in viewer.yaml
- Allows activating a layer (sets it as the paint target)
- Shows class palette with colour swatches
- Allows adding new classes interactively
- Classes are shared across layers (global class list)
"""
from __future__ import annotations

from typing import Optional

import napari
import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from biovision_napari.io.yaml_schema import LabelLayer, save_viewer_yaml
from biovision_napari.state.project_state import ProjectState


class ClassPaletteItem(QWidget):
    """A coloured swatch + class name inside the class list."""

    def __init__(self, class_id: int, class_name: str, color: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        swatch = QLabel()
        swatch.setFixedSize(16, 16)
        swatch.setStyleSheet(f"background-color: {color}; border: 1px solid #333;")
        layout.addWidget(swatch)

        lbl = QLabel(f"[{class_id}] {class_name}")
        layout.addWidget(lbl)
        layout.addStretch()


class LabelControls(QWidget):
    """Dock widget for label layer and class management."""

    layer_activated = Signal(str)    # emits layer name
    class_selected = Signal(int)     # emits class id

    def __init__(
        self,
        viewer: napari.Viewer,
        state: ProjectState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._state = state
        self._label_layers: list[LabelLayer] = []

        self._build_ui()
        self._state.config_changed.connect(self._on_config_changed)
        self._state.sample_changed.connect(self._on_sample_changed)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Layer selector
        layer_group = QGroupBox("Label Layers")
        layer_layout = QVBoxLayout(layer_group)
        self._layer_list = QListWidget()
        self._layer_list.currentRowChanged.connect(self._on_layer_selected)
        layer_layout.addWidget(self._layer_list)
        layout.addWidget(layer_group)

        # Class palette
        class_group = QGroupBox("Classes")
        class_layout = QVBoxLayout(class_group)
        self._class_list = QListWidget()
        self._class_list.currentRowChanged.connect(self._on_class_selected)
        class_layout.addWidget(self._class_list)

        btn_row = QHBoxLayout()
        btn_add_class = QPushButton("+ Add Class")
        btn_add_class.clicked.connect(self._add_class_interactive)
        btn_row.addWidget(btn_add_class)
        btn_row.addStretch()
        class_layout.addLayout(btn_row)
        layout.addWidget(class_group)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_config_changed(self, config) -> None:
        if config is None:
            return
        self._label_layers = config.label_layers
        self._refresh_layer_list()

    def _on_sample_changed(self, sample_id: str) -> None:
        self._create_label_layers_in_viewer()

    def _on_layer_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._label_layers):
            return
        layer_def = self._label_layers[row]
        self._activate_layer(layer_def.name)
        self._refresh_class_list(layer_def)
        self.layer_activated.emit(layer_def.name)

    def _on_class_selected(self, row: int) -> None:
        if row < 0:
            return
        # Find active layer
        layer_row = self._layer_list.currentRow()
        if layer_row < 0 or layer_row >= len(self._label_layers):
            return
        self.class_selected.emit(row)
        # Set the label value on the napari layer
        self._set_active_label_value(row)

    # ------------------------------------------------------------------
    # Viewer interaction
    # ------------------------------------------------------------------

    def _create_label_layers_in_viewer(self) -> None:
        """
        Create (or reset) one Labels layer in the viewer per YAML label_layer.
        Layers are created with empty (zeros) arrays matching the current image shape.
        """
        # Determine shape from first image layer
        shape = self._get_image_shape()
        if shape is None:
            return

        for layer_def in self._label_layers:
            existing = [
                l for l in self._viewer.layers
                if l.name == layer_def.name
            ]
            if existing:
                continue  # already present

            label_data = np.zeros(shape, dtype=np.uint32)
            self._viewer.add_labels(
                label_data,
                name=layer_def.name,
                opacity=layer_def.opacity,
            )

    def _activate_layer(self, layer_name: str) -> None:
        """Make the named label layer the active selection in the viewer."""
        for layer in self._viewer.layers:
            if layer.name == layer_name:
                self._viewer.layers.selection.active = layer
                break

    def _set_active_label_value(self, class_id: int) -> None:
        """Set the paint label value on the active Labels layer."""
        active = self._viewer.layers.selection.active
        if active is not None and hasattr(active, "selected_label"):
            active.selected_label = class_id

    def _get_image_shape(self):
        """Return the shape of the first Image layer, or None."""
        for layer in self._viewer.layers:
            if hasattr(layer, "data") and not hasattr(layer, "selected_label"):
                return layer.data.shape
        return None

    # ------------------------------------------------------------------
    # UI refresh
    # ------------------------------------------------------------------

    def _refresh_layer_list(self) -> None:
        self._layer_list.clear()
        for layer_def in self._label_layers:
            item = QListWidgetItem(layer_def.name)
            item.setForeground(QColor(layer_def.color))
            self._layer_list.addItem(item)
        if self._label_layers:
            self._layer_list.setCurrentRow(0)

    def _refresh_class_list(self, layer_def: LabelLayer) -> None:
        self._class_list.clear()
        palette = _generate_class_colors(len(layer_def.classes))
        for class_id, (class_name, color) in enumerate(zip(layer_def.classes, palette)):
            item = QListWidgetItem()
            item.setSizeHint(ClassPaletteItem(class_id, class_name, color).sizeHint())
            self._class_list.addItem(item)
            widget = ClassPaletteItem(class_id, class_name, color)
            self._class_list.setItemWidget(item, widget)
        if layer_def.classes:
            self._class_list.setCurrentRow(1 if len(layer_def.classes) > 1 else 0)

    # ------------------------------------------------------------------
    # Interactive class addition
    # ------------------------------------------------------------------

    def _add_class_interactive(self) -> None:
        layer_row = self._layer_list.currentRow()
        if layer_row < 0 or layer_row >= len(self._label_layers):
            return

        dialog = _AddClassDialog(self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_class = dialog.class_name()
        if not new_class:
            return

        # Add to YAML config and persist
        layer_def = self._label_layers[layer_row]
        if new_class not in layer_def.classes:
            layer_def.classes.append(new_class)
            if self._state.config is not None and self._state.yaml_path is not None:
                save_viewer_yaml(self._state.config, self._state.yaml_path)

        self._refresh_class_list(layer_def)


class _AddClassDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Class")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Class name:"))
        self._edit = QLineEdit()
        layout.addWidget(self._edit)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def class_name(self) -> str:
        return self._edit.text().strip()


def _generate_class_colors(n: int) -> list[str]:
    """Generate n visually distinct hex colors."""
    if n == 0:
        return []
    # Fixed background color, then evenly spaced hues
    colors = ["#222222"]  # background = dark
    for i in range(1, n):
        hue = int((i / max(n - 1, 1)) * 300)  # 0–300 deg to avoid red wrap
        color = QColor.fromHsv(hue, 200, 220)
        colors.append(color.name())
    return colors[:n]
