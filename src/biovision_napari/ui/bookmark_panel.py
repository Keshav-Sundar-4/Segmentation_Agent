"""
Bookmark panel: capture and restore sample_id + Z + T + optional note.
Bookmarks are persisted directly in viewer.yaml under the 'bookmarks' key.
"""
from __future__ import annotations

from typing import Optional

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from biovision_napari.io.yaml_schema import Bookmark, save_viewer_yaml
from biovision_napari.state.project_state import ProjectState


class BookmarkPanel(QWidget):

    def __init__(
        self,
        viewer: napari.Viewer,
        state: ProjectState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._state = state
        self._build_ui()
        self._state.config_changed.connect(self._on_config_changed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(QLabel("Bookmarks"))

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("+ Bookmark here")
        btn_add.clicked.connect(self._add_bookmark)
        btn_row.addWidget(btn_add)

        btn_del = QPushButton("Delete")
        btn_del.clicked.connect(self._delete_selected)
        btn_row.addWidget(btn_del)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def _on_config_changed(self, config) -> None:
        self._refresh()

    def _refresh(self) -> None:
        cfg = self._state.config
        self._list.clear()
        if cfg is None:
            return
        for bm in cfg.bookmarks:
            text = f"[{bm.sample_id}] Z={bm.z} T={bm.t}"
            if bm.note:
                text += f" — {bm.note}"
            self._list.addItem(QListWidgetItem(text))

    def _add_bookmark(self) -> None:
        cfg = self._state.config
        sample_id = self._state.active_sample
        if cfg is None or sample_id is None:
            return

        z, t = self._current_zt()
        note, ok = QInputDialog.getText(self, "Add Bookmark", "Note (optional):")
        if not ok:
            return

        bm = Bookmark(sample_id=sample_id, z=z, t=t, note=note.strip())
        cfg.bookmarks.append(bm)
        if self._state.yaml_path:
            save_viewer_yaml(cfg, self._state.yaml_path)
        self._refresh()

    def _delete_selected(self) -> None:
        cfg = self._state.config
        row = self._list.currentRow()
        if cfg is None or row < 0 or row >= len(cfg.bookmarks):
            return
        cfg.bookmarks.pop(row)
        if self._state.yaml_path:
            save_viewer_yaml(cfg, self._state.yaml_path)
        self._refresh()

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        cfg = self._state.config
        row = self._list.row(item)
        if cfg is None or row < 0 or row >= len(cfg.bookmarks):
            return
        bm = cfg.bookmarks[row]
        # Restore sample
        self._state.set_active_sample(bm.sample_id)
        # Restore Z/T
        self._goto_zt(bm.z, bm.t)

    def _current_zt(self) -> tuple[int, int]:
        """Read current Z and T from viewer dims using axis_order."""
        step = list(self._viewer.dims.current_step)
        axis_order = getattr(self._viewer, "_biovision_axis_order", "TZYX")
        z = _get_axis_value(step, axis_order, "Z")
        t = _get_axis_value(step, axis_order, "T")
        return z, t

    def _goto_zt(self, z: int, t: int) -> None:
        step = list(self._viewer.dims.current_step)
        axis_order = getattr(self._viewer, "_biovision_axis_order", "TZYX")
        _set_axis_value(step, axis_order, "Z", z)
        _set_axis_value(step, axis_order, "T", t)
        self._viewer.dims.current_step = tuple(step)


def _get_axis_value(step: list[int], axis_order: str, axis: str) -> int:
    try:
        idx = axis_order.upper().index(axis.upper())
        return step[idx] if idx < len(step) else 0
    except ValueError:
        return 0


def _set_axis_value(step: list[int], axis_order: str, axis: str, value: int) -> None:
    try:
        idx = axis_order.upper().index(axis.upper())
        if idx < len(step):
            step[idx] = value
    except ValueError:
        pass
