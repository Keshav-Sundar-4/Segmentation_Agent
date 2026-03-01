"""
agent_panel.py — Human-in-the-Loop agent panel for BioVision.

Layout (top → bottom)
─────────────────────
  [INPUT CONFIGURATION]
    Images folder    [path edit]   [Browse…]
    Metadata YAML    [path edit]   [Browse…]
    Anthropic Key    [password edit]

  [▶ Run Agent]  [■ Stop]   Status: …

  [AGENT LOG]
    scrollable monospace output

  [HUMAN REVIEW]  ← hidden until mini-batch is ready
    Technique name + description
    napari layers info label
    [✓ Accept]  [✗ Reject]
    Feedback (optional) text field

  [FULL RUN PROGRESS]  ← hidden until Accept pressed
    Progress bar
    Output path label
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from biovision_napari.state.project_state import ProjectState

logger = logging.getLogger(__name__)

# Layer name prefixes used for HITL review layers (easy to identify & clean up)
_ORIG_PREFIX = "hitl_original_"
_PROC_PREFIX = "hitl_processed_"
_MAX_REVIEW_PAIRS = 3   # max image pairs shown in the viewer during review


class AgentPanel(QWidget):
    """
    HITL agent panel.  Provides:
      • Input configuration (folder, YAML, API key)
      • Async agent trigger (never blocks the Qt event loop)
      • Live log streaming
      • Human review gate (Accept / Reject + optional feedback)
      • Full-run progress display
    """

    agent_finished = Signal()   # emitted when the agent cycle completes

    def __init__(
        self,
        viewer,                          # napari.Viewer — used to add review layers
        state: ProjectState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._state = state
        self._worker = None
        self._build_ui()
        self._state.config_changed.connect(self._on_config_changed)

    # ──────────────────────────────────────────────────────────────────────────
    # UI construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)

        root.addWidget(self._build_config_group())
        root.addLayout(self._build_controls_row())
        root.addWidget(self._build_log_area())
        root.addWidget(self._build_review_section())
        root.addWidget(self._build_progress_section())

    # ── Configuration group ───────────────────────────────────────────────────

    def _build_config_group(self) -> QGroupBox:
        grp = QGroupBox("Configuration")
        from qtpy.QtWidgets import QGridLayout
        grid = QGridLayout(grp)
        grid.setSpacing(4)

        # Row 0 — images folder
        grid.addWidget(QLabel("Images:"), 0, 0, Qt.AlignRight)
        self._edit_folder = QLineEdit()
        self._edit_folder.setPlaceholderText("Select the folder containing input images…")
        grid.addWidget(self._edit_folder, 0, 1)
        btn_folder = QPushButton("Browse…")
        btn_folder.setFixedWidth(70)
        btn_folder.clicked.connect(self._browse_folder)
        grid.addWidget(btn_folder, 0, 2)

        # Row 1 — metadata YAML
        grid.addWidget(QLabel("Metadata YAML:"), 1, 0, Qt.AlignRight)
        self._edit_yaml = QLineEdit()
        self._edit_yaml.setPlaceholderText("Select metadata YAML file…")
        grid.addWidget(self._edit_yaml, 1, 1)
        btn_yaml = QPushButton("Browse…")
        btn_yaml.setFixedWidth(70)
        btn_yaml.clicked.connect(self._browse_yaml)
        grid.addWidget(btn_yaml, 1, 2)

        # Row 2 — API key
        grid.addWidget(QLabel("Anthropic API Key:"), 2, 0, Qt.AlignRight)
        self._edit_apikey = QLineEdit()
        self._edit_apikey.setEchoMode(QLineEdit.Password)
        self._edit_apikey.setPlaceholderText("sk-ant-…  (stored in memory only, never saved)")
        grid.addWidget(self._edit_apikey, 2, 1, 1, 2)

        grid.setColumnStretch(1, 1)
        return grp

    # ── Run / Stop controls row ───────────────────────────────────────────────

    def _build_controls_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self._btn_run = QPushButton("▶  Run Agent")
        self._btn_run.setMinimumHeight(28)
        self._btn_run.clicked.connect(self._run_agent)
        row.addWidget(self._btn_run)

        self._btn_stop = QPushButton("■  Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.setMinimumHeight(28)
        self._btn_stop.clicked.connect(self._stop_agent)
        row.addWidget(self._btn_stop)

        row.addStretch()

        self._lbl_status = QLabel("Status: idle")
        self._lbl_status.setStyleSheet("color: #888; font-size: 10px;")
        row.addWidget(self._lbl_status)

        return row

    # ── Live log area ─────────────────────────────────────────────────────────

    def _build_log_area(self) -> QPlainTextEdit:
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setStyleSheet("font-family: monospace; font-size: 10px;")
        self._log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return self._log

    # ── Human review section ──────────────────────────────────────────────────

    def _build_review_section(self) -> QWidget:
        self._review_widget = QWidget()
        layout = QVBoxLayout(self._review_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Human Review — Evaluate the Mini-Batch")
        inner = QVBoxLayout(grp)

        self._lbl_technique_name = QLabel()
        self._lbl_technique_name.setWordWrap(True)
        self._lbl_technique_name.setStyleSheet("font-weight: bold; font-size: 11px;")
        inner.addWidget(self._lbl_technique_name)

        self._lbl_technique_desc = QLabel()
        self._lbl_technique_desc.setWordWrap(True)
        self._lbl_technique_desc.setStyleSheet("color: #ccc; font-size: 10px;")
        inner.addWidget(self._lbl_technique_desc)

        self._lbl_layers_info = QLabel(
            f"Original and preprocessed images have been added as napari layers\n"
            f"(prefixed '{_ORIG_PREFIX}' and '{_PROC_PREFIX}') — toggle visibility to compare."
        )
        self._lbl_layers_info.setWordWrap(True)
        self._lbl_layers_info.setStyleSheet("color: #aad4ff; font-size: 10px; padding-top: 4px;")
        inner.addWidget(self._lbl_layers_info)

        btn_row = QHBoxLayout()
        self._btn_accept = QPushButton("✓  Accept — run on full dataset")
        self._btn_accept.setMinimumHeight(30)
        self._btn_accept.setStyleSheet(
            "QPushButton { background-color: #2a5c2a; color: white; font-weight: bold; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a7c3a; }"
        )
        self._btn_accept.clicked.connect(self._accept)
        btn_row.addWidget(self._btn_accept)

        self._btn_reject = QPushButton("✗  Reject — try a different technique")
        self._btn_reject.setMinimumHeight(30)
        self._btn_reject.setStyleSheet(
            "QPushButton { background-color: #5c2a2a; color: white; border-radius: 4px; }"
            "QPushButton:hover { background-color: #7c3a3a; }"
        )
        self._btn_reject.clicked.connect(self._reject)
        btn_row.addWidget(self._btn_reject)
        inner.addLayout(btn_row)

        inner.addWidget(QLabel("Feedback (optional — helps the agent improve):"))
        self._edit_feedback = QLineEdit()
        self._edit_feedback.setPlaceholderText(
            "e.g. 'too much noise removed', 'contrast too high', 'try anisotropic diffusion'…"
        )
        inner.addWidget(self._edit_feedback)

        layout.addWidget(grp)
        self._review_widget.hide()
        return self._review_widget

    # ── Full-run progress section ─────────────────────────────────────────────

    def _build_progress_section(self) -> QWidget:
        self._progress_widget = QWidget()
        layout = QVBoxLayout(self._progress_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Full Dataset Execution")
        inner = QVBoxLayout(grp)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)   # indeterminate until done
        self._progress_bar.setTextVisible(False)
        inner.addWidget(self._progress_bar)

        self._lbl_output = QLabel("Processing images…")
        self._lbl_output.setWordWrap(True)
        self._lbl_output.setStyleSheet("font-size: 10px;")
        inner.addWidget(self._lbl_output)

        layout.addWidget(grp)
        self._progress_widget.hide()
        return self._progress_widget

    # ──────────────────────────────────────────────────────────────────────────
    # File-browser helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _browse_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select image folder", "")
        if path:
            self._edit_folder.setText(path)

    def _browse_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata YAML", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if path:
            self._edit_yaml.setText(path)

    # ──────────────────────────────────────────────────────────────────────────
    # Config-change hook  (pre-fills fields from viewer.yaml if present)
    # ──────────────────────────────────────────────────────────────────────────

    def _on_config_changed(self, config) -> None:
        if config is None:
            return
        # Pre-fill folder if project config provides a dataset root
        if self._edit_folder.text() == "":
            try:
                root = config.paths.dataset_root
                if root:
                    self._edit_folder.setText(str(root))
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────────────────
    # Agent lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def _run_agent(self) -> None:
        folder = self._edit_folder.text().strip()
        yaml_path = self._edit_yaml.text().strip()
        api_key = self._edit_apikey.text().strip()

        # Validate
        missing = []
        if not folder:
            missing.append("Images folder")
        if not yaml_path:
            missing.append("Metadata YAML")
        if not api_key:
            missing.append("Anthropic API Key")
        if missing:
            QMessageBox.warning(self, "Missing inputs", "Please fill in:\n• " + "\n• ".join(missing))
            return
        if not Path(folder).is_dir():
            QMessageBox.warning(self, "Invalid folder", f"Directory not found:\n{folder}")
            return
        if not Path(yaml_path).is_file():
            QMessageBox.warning(self, "Invalid YAML", f"File not found:\n{yaml_path}")
            return

        # Reset UI
        self._log.clear()
        self._review_widget.hide()
        self._progress_widget.hide()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._set_status("running…")

        output_root = str(Path(folder) / "outputs")

        from biovision_napari.workers.hitl_worker import hitl_worker
        self._worker = hitl_worker(
            input_folder=folder,
            metadata_path=yaml_path,
            api_key=api_key,
            sample_size=5,
            output_root=output_root,
        )
        self._worker.yielded.connect(self._on_yield)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.errored.connect(self._on_worker_error)
        self._worker.start()

    def _stop_agent(self) -> None:
        if self._worker is not None:
            self._worker.quit()
        self._review_widget.hide()
        self._progress_widget.hide()

    # ──────────────────────────────────────────────────────────────────────────
    # Worker event dispatch
    # ──────────────────────────────────────────────────────────────────────────

    def _on_yield(self, event: tuple) -> None:
        kind, payload = event

        if kind == "log":
            self._append_log(str(payload))

        elif kind == "error":
            self._append_log(f"ERROR: {payload}", error=True)
            self._set_idle("error")

        elif kind == "review":
            self._show_review(payload)

        elif kind == "done":
            self._on_done(str(payload))

    def _on_worker_finished(self) -> None:
        self._set_idle("done")
        self.agent_finished.emit()

    def _on_worker_error(self, exc: Exception) -> None:
        self._append_log(f"Worker error: {exc}", error=True)
        self._set_idle("error")

    # ──────────────────────────────────────────────────────────────────────────
    # Log helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _append_log(self, text: str, error: bool = False) -> None:
        if error:
            self._log.appendHtml(f'<span style="color:#ff7777;">▸ {text}</span>')
        else:
            self._log.appendPlainText(f"▸ {text}")
        # Auto-scroll to bottom
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log.setTextCursor(cursor)

    # ──────────────────────────────────────────────────────────────────────────
    # Review section
    # ──────────────────────────────────────────────────────────────────────────

    def _show_review(self, payload: dict) -> None:
        technique_name = payload.get("technique_name", "Unknown technique")
        technique_desc = payload.get("technique_description", "")
        results = payload.get("mini_batch_results", [])

        self._lbl_technique_name.setText(f"Technique: {technique_name}")
        self._lbl_technique_desc.setText(technique_desc)
        self._edit_feedback.clear()
        self._review_widget.show()
        self._set_status("awaiting your review")

        # Load original + processed pairs into napari
        self._load_review_layers(results, technique_name)

    def _load_review_layers(self, results: list, technique_name: str) -> None:
        """Add original and preprocessed images as napari layers for comparison."""
        # Remove any stale review layers from a previous round
        stale = [
            lyr for lyr in self._viewer.layers
            if lyr.name.startswith(_ORIG_PREFIX) or lyr.name.startswith(_PROC_PREFIX)
        ]
        for lyr in stale:
            try:
                self._viewer.layers.remove(lyr)
            except Exception:
                pass

        pairs_added = 0
        short_name = technique_name[:20].replace(" ", "_")

        for i, result in enumerate(results[:_MAX_REVIEW_PAIRS]):
            orig_path = result.get("original_path")
            proc_path = result.get("processed_path")

            orig_arr = _load_image(orig_path)
            if orig_arr is not None:
                self._viewer.add_image(
                    orig_arr,
                    name=f"{_ORIG_PREFIX}{i}",
                    colormap="gray",
                    blending="additive",
                    visible=(i == 0),
                )
                pairs_added += 1

            if proc_path:
                proc_arr = _load_image(proc_path)
                if proc_arr is not None:
                    self._viewer.add_image(
                        proc_arr,
                        name=f"{_PROC_PREFIX}{i}_{short_name}",
                        colormap="green",
                        blending="additive",
                        visible=(i == 0),
                    )

        if pairs_added:
            info = (
                f"{pairs_added} pair(s) added as napari layers "
                f"('{_ORIG_PREFIX}*' = original, '{_PROC_PREFIX}*' = processed).\n"
                "Toggle visibility in the Layers panel to compare."
            )
        else:
            info = "Could not load preview images — check the log for errors."
        self._lbl_layers_info.setText(info)

    # ──────────────────────────────────────────────────────────────────────────
    # Accept / Reject
    # ──────────────────────────────────────────────────────────────────────────

    def _accept(self) -> None:
        if self._worker is None:
            return
        self._review_widget.hide()
        self._progress_widget.show()
        self._progress_bar.setRange(0, 0)   # indeterminate spinner
        self._set_status("running full dataset…")
        self._append_log("─" * 50)
        self._append_log("Accepted.  Running on full dataset…")
        self._worker.send({"action": "accept"})

    def _reject(self) -> None:
        if self._worker is None:
            return
        feedback = self._edit_feedback.text().strip()
        self._review_widget.hide()
        self._append_log("─" * 50)
        self._append_log(f"Rejected.  Feedback: {feedback or '(none)'}.")
        self._append_log("Researching a different technique…")
        self._set_status("rethinking technique…")
        self._worker.send({"action": "reject", "feedback": feedback})

    # ──────────────────────────────────────────────────────────────────────────
    # Full-run completion
    # ──────────────────────────────────────────────────────────────────────────

    def _on_done(self, output_dir: str) -> None:
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(100)
        self._lbl_output.setText(f"Output saved to:\n{output_dir}")
        self._set_status("done ✓")
        self._append_log(f"Complete!  Output directory: {output_dir}")
        self._set_idle("done")

    # ──────────────────────────────────────────────────────────────────────────
    # Status / idle helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _set_status(self, text: str) -> None:
        self._lbl_status.setText(f"Status: {text}")

    def _set_idle(self, reason: str = "idle") -> None:
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._worker = None
        if reason not in ("done",):
            self._set_status(reason)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_image(path: Optional[str]) -> Optional[np.ndarray]:
    """Load an image file to a numpy array, or return None on failure."""
    if not path:
        return None
    try:
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            return None
        # Convert BGR → RGB for colour images so napari shows correct colours
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = arr[..., ::-1].copy()
        return arr
    except Exception as exc:
        logger.warning("Could not load review image %s: %s", path, exc)
        return None
