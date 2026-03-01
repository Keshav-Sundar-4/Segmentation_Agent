"""
Agent execution panel.
Provides a "Run Agent" button, shows live stdout/stderr output,
and triggers a model-list refresh when the agent finishes.
"""
from __future__ import annotations

from typing import Optional

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QColor, QTextCursor
from qtpy.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from biovision_napari.state.project_state import ProjectState
from biovision_napari.workers.agent_worker import run_agent_worker


class AgentPanel(QWidget):

    agent_finished = Signal()  # emits when agent process completes

    def __init__(
        self,
        state: ProjectState,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._worker = None
        self._build_ui()
        self._state.config_changed.connect(self._on_config_changed)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        self._lbl_status = QLabel("Agent: idle")
        header.addWidget(self._lbl_status)
        header.addStretch()

        self._btn_run = QPushButton("Run Agent")
        self._btn_run.clicked.connect(self._run_agent)
        header.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_agent)
        header.addWidget(self._btn_stop)

        layout.addLayout(header)

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(2000)
        self._output.setStyleSheet("font-family: monospace; font-size: 10px;")
        layout.addWidget(self._output)

    def _on_config_changed(self, config) -> None:
        pass  # command is read at run time

    def _run_agent(self) -> None:
        cfg = self._state.config
        if cfg is None:
            return

        self._output.clear()
        self._lbl_status.setText("Agent: running...")
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._worker = run_agent_worker(
            command=cfg.agent.command,
            working_dir=cfg.agent.working_dir,
        )
        self._worker.yielded.connect(self._on_output_line)
        self._worker.finished.connect(self._on_agent_finished)
        self._worker.errored.connect(self._on_agent_error)
        self._worker.start()

    def _stop_agent(self) -> None:
        if self._worker is not None:
            self._worker.quit()

    def _on_output_line(self, line_err: tuple) -> None:
        line, is_err = line_err
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if is_err:
            self._output.appendHtml(f'<span style="color:#ff6666;">{line}</span>')
        else:
            self._output.appendPlainText(line)

    def _on_agent_finished(self) -> None:
        self._lbl_status.setText("Agent: done")
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._worker = None
        self.agent_finished.emit()

    def _on_agent_error(self, exc: Exception) -> None:
        self._lbl_status.setText(f"Agent: error — {exc}")
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._worker = None
