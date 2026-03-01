"""
Dataset browser dock widget.
Shows a table of samples with columns:
  sample_id | status | modality | dims | has_gt_mask | models_available
Clicking a row loads that sample into the viewer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from biovision_napari.io.sample_discovery import (
    VALID_STATUSES,
    SampleInfo,
    discover_samples,
    write_sample_status,
)
from biovision_napari.state.project_state import ProjectState


COLUMNS = ["Sample ID", "Status", "Modality", "Dims", "Has GT Mask", "Models"]

STATUS_COLORS = {
    "unlabeled": "#555555",
    "in_progress": "#cc8800",
    "done": "#228822",
    "reviewed": "#2244cc",
}


class DatasetBrowser(QWidget):
    """Dock widget for browsing and selecting samples."""

    sample_selected = Signal(str)  # emits sample_id

    def __init__(
        self,
        state: ProjectState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._samples: list[SampleInfo] = []

        self._build_ui()
        self._state.config_changed.connect(self._on_config_changed)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar row
        toolbar = QHBoxLayout()
        self._lbl_project = QLabel("No project loaded")
        self._lbl_project.setStyleSheet("font-weight: bold;")
        toolbar.addWidget(self._lbl_project)
        toolbar.addStretch()

        btn_refresh = QPushButton("Refresh")
        btn_refresh.setFixedWidth(70)
        btn_refresh.clicked.connect(self._refresh)
        toolbar.addWidget(btn_refresh)
        layout.addLayout(toolbar)

        # Table
        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self._table.cellClicked.connect(self._on_row_clicked)
        layout.addWidget(self._table)

        # Status editor row
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Set status:"))
        self._status_combo = QComboBox()
        self._status_combo.addItems(sorted(VALID_STATUSES))
        status_row.addWidget(self._status_combo)
        btn_set_status = QPushButton("Apply")
        btn_set_status.clicked.connect(self._apply_status)
        status_row.addWidget(btn_set_status)
        status_row.addStretch()
        layout.addLayout(status_row)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_config_changed(self, config) -> None:
        name = config.project.name if config else "—"
        self._lbl_project.setText(f"Project: {name}")
        self._refresh()

    def _refresh(self) -> None:
        cfg = self._state.config
        if cfg is None:
            return
        self._samples = discover_samples(
            dataset_root=cfg.paths.dataset_root,
            status_file=cfg.samples.status_file,
            masks_root=cfg.paths.masks,
        )
        self._populate_table()

    def _populate_table(self) -> None:
        self._table.setRowCount(0)
        for row_idx, sample in enumerate(self._samples):
            self._table.insertRow(row_idx)
            row_data = sample.to_display_row()
            for col_idx, text in enumerate(row_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # Colour-code status column
                if col_idx == 1:
                    color = STATUS_COLORS.get(text, "#ffffff")
                    item.setForeground(QColor(color))
                self._table.setItem(row_idx, col_idx, item)

    def _on_row_clicked(self, row: int, _col: int) -> None:
        if 0 <= row < len(self._samples):
            sample_id = self._samples[row].sample_id
            self._state.set_active_sample(sample_id)
            self.sample_selected.emit(sample_id)

    def _on_row_double_clicked(self, row: int, _col: int) -> None:
        self._on_row_clicked(row, _col)

    def _apply_status(self) -> None:
        selected = self._table.selectedItems()
        if not selected:
            return
        row = self._table.currentRow()
        if row < 0 or row >= len(self._samples):
            return
        sample = self._samples[row]
        new_status = self._status_combo.currentText()
        cfg = self._state.config
        write_sample_status(
            sample_path=sample.path,
            status_file=cfg.samples.status_file if cfg else "status.json",
            status=new_status,
            modality=sample.modality,
            dims=sample.dims,
        )
        sample.status = new_status
        self._populate_table()
