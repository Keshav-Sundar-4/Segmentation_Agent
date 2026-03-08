"""
Agent panel — runs the BioVision preprocessing pipeline from the Napari UI.

User flow
---------
1. Type or browse to a metadata YAML file.
2. Type or browse to an input image folder.
3. Choose backend: Claude or Local (Ollama).
4. If Claude: paste / confirm Anthropic API key.
   If Ollama:  pick a model from the auto-detected list (or type one).
5. Optionally configure sandbox mode (Docker) and max retries.
6. Click Run.

Design decisions
----------------
* Paths are stored as absolute strings; image folders are NEVER copied.
* API keys are session-only — never written to disk or logged.
* Non-secret settings (provider, model, last paths) are persisted to
  ~/.biovision/agent_prefs.json between sessions.
* The agent is run via run_biovision_agent_worker which calls
  Agent.main.run_pipeline_stream() directly — no shell command needed.
* Drag-and-drop accepts YAML files and/or directories in any combination.
* llm_config_ready signal is emitted when pipeline starts so the LLM chat
  widget can synchronise to the same provider/model/key.
* input_dir_changed signal is emitted whenever the image folder changes so
  the main widget can auto-load images into the viewer.
"""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from napari.qt.threading import thread_worker

from biovision_napari.state.project_state import ProjectState
from biovision_napari.workers.agent_worker import run_biovision_agent_worker

# ---------------------------------------------------------------------------
# Preferences helpers (non-secret settings only)
# ---------------------------------------------------------------------------

_PREFS_PATH = Path.home() / ".biovision" / "agent_prefs.json"

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ndpi", ".svs"}


def _load_prefs() -> dict:
    try:
        if _PREFS_PATH.exists():
            with open(_PREFS_PATH, encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return {}


def _save_prefs(prefs: dict) -> None:
    try:
        _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PREFS_PATH, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh, indent=2)
    except Exception:
        pass


def _count_images(folder: str) -> int:
    try:
        return sum(
            1 for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
    except Exception:
        return 0


def _ts() -> str:
    """Return a compact HH:MM:SS timestamp for log lines."""
    return datetime.datetime.now().strftime("%H:%M:%S")


# ---------------------------------------------------------------------------
# Ollama model discovery (runs in background)
# ---------------------------------------------------------------------------

_OLLAMA_DEFAULT_URL = "http://localhost:11434"


@thread_worker
def _fetch_ollama_models_worker(base_url: str):
    """Yield list[str] of model names or raise on failure."""
    url = base_url.rstrip("/") + "/api/tags"
    with urllib.request.urlopen(url, timeout=4) as resp:
        data = json.loads(resp.read().decode())
    models = [m["name"] for m in data.get("models", [])]
    yield models


# ---------------------------------------------------------------------------
# Resource monitor (psutil + optional nvidia-smi)
# ---------------------------------------------------------------------------

def _read_resources() -> dict:
    """Return {cpu, ram_mb, gpu_pct, gpu_mem_mb} — best-effort, no exceptions."""
    result = {"cpu": 0.0, "ram_mb": 0, "gpu_pct": None, "gpu_mem_mb": None}
    try:
        import psutil
        result["cpu"] = psutil.cpu_percent(interval=None)
        result["ram_mb"] = int(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        parts = out.split(",")
        result["gpu_pct"] = int(parts[0].strip())
        result["gpu_mem_mb"] = int(parts[1].strip())
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Node name → human-readable status
# ---------------------------------------------------------------------------

_NODE_LABELS = {
    "planner":          "Planning preprocessing steps…",
    "coder":            "Generating Python script…",
    "sandbox_executor": "Validating script on sample images…",
    "local_executor":   "Running on full dataset…",
    "terminal_failure": "Pipeline failed — max retries exhausted.",
}


def _node_label(node_name: str) -> str:
    return _NODE_LABELS.get(node_name, f"[{node_name}]")


# ---------------------------------------------------------------------------
# Drop-zone label widget
# ---------------------------------------------------------------------------

class _DropZone(QLabel):
    """A label that lights up when something valid is dragged over it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Drop metadata YAML or image folder here")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._normal_style = (
            "border: 2px dashed #555; border-radius: 6px; "
            "color: #888; font-size: 10px; padding: 8px;"
        )
        self._hover_style = (
            "border: 2px dashed #44aaff; border-radius: 6px; "
            "color: #44aaff; font-size: 10px; padding: 8px;"
        )
        self.setStyleSheet(self._normal_style)

    def set_hover(self, active: bool) -> None:
        self.setStyleSheet(self._hover_style if active else self._normal_style)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class AgentPanel(QWidget):
    """
    Agent panel — self-contained UI for running the BioVision pipeline.
    Accepts file/folder drops and exposes provider/model selection.
    """

    agent_finished = Signal()
    # Emitted when the pipeline starts with the active LLM config so other
    # widgets (e.g. LLMChatWidget) can synchronise to the same connection.
    llm_config_ready = Signal(str, str, str, str)  # provider, model, api_key, base_url
    # Emitted whenever the image folder changes so main_widget can auto-load.
    input_dir_changed = Signal(str)

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(self, state: ProjectState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._worker = None
        self._ollama_worker = None

        # Activity-indicator state (pulsing spinner shown while pipeline runs).
        self._spinner_frames: list[str] = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._spinner_frame:  int = 0
        self._activity_text:  str = ""
        self._retry_count:    int = 0
        self._activity_current: Optional[QLabel] = None
        self._activity_layout:  Optional[QVBoxLayout] = None
        self._activity_frame:   Optional[QFrame] = None
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._pulse_spinner)

        # Resource monitor timer (polls CPU/RAM/GPU while running).
        self._resource_timer = QTimer(self)
        self._resource_timer.timeout.connect(self._update_resources)
        self._resource_lbl: Optional[QLabel] = None

        # Current inputs (paths stored as absolute strings, never copied).
        self._yaml_path: str = ""
        self._input_dir: str = ""
        self._output_dir: str = ""

        # Accept drops on the whole panel.
        self.setAcceptDrops(True)

        self._build_ui()
        self._load_prefs()
        self._validate()

        # Auto-detect Ollama models in the background on startup.
        QTimer.singleShot(500, self._detect_ollama)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(6)

        outer.addWidget(self._build_input_group())
        outer.addWidget(self._build_model_group())
        outer.addWidget(self._build_sandbox_group())
        outer.addWidget(self._build_run_bar())
        outer.addWidget(self._build_resource_bar())
        outer.addWidget(self._build_activity_widget())
        outer.addWidget(self._build_log())

    # ── Inputs group ──────────────────────────────────────────────────────────

    def _build_input_group(self) -> QGroupBox:
        box = QGroupBox("Inputs")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # YAML row — editable field so user can type/paste a path directly
        yaml_row = QHBoxLayout()
        yaml_row.addWidget(QLabel("Metadata YAML:"))
        self._edit_yaml = QLineEdit()
        self._edit_yaml.setPlaceholderText("Path to metadata.yaml…")
        self._edit_yaml.setStyleSheet("font-size: 10px;")
        self._edit_yaml.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._edit_yaml.editingFinished.connect(self._on_yaml_edited)
        yaml_row.addWidget(self._edit_yaml, stretch=1)
        btn_yaml = QPushButton("Browse…")
        btn_yaml.setFixedWidth(72)
        btn_yaml.clicked.connect(self._browse_yaml)
        yaml_row.addWidget(btn_yaml)
        layout.addLayout(yaml_row)

        # Image folder row — editable field
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Image folder:"))
        self._edit_dir = QLineEdit()
        self._edit_dir.setPlaceholderText("Path to image folder…")
        self._edit_dir.setStyleSheet("font-size: 10px;")
        self._edit_dir.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._edit_dir.editingFinished.connect(self._on_dir_edited)
        dir_row.addWidget(self._edit_dir, stretch=1)
        btn_dir = QPushButton("Browse…")
        btn_dir.setFixedWidth(72)
        btn_dir.clicked.connect(self._browse_dir)
        dir_row.addWidget(btn_dir)
        layout.addLayout(dir_row)

        # Output folder row (derived, read-only)
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output folder:"))
        self._lbl_out = QLabel("—")
        self._lbl_out.setStyleSheet("color: #666; font-size: 10px;")
        self._lbl_out.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        out_row.addWidget(self._lbl_out, stretch=1)
        layout.addLayout(out_row)

        # Drop zone
        self._drop_zone = _DropZone()
        layout.addWidget(self._drop_zone)

        # Validation status
        self._lbl_input_status = QLabel("")
        self._lbl_input_status.setStyleSheet("font-size: 10px; color: #aaa;")
        layout.addWidget(self._lbl_input_status)

        return box

    # ── Model group ───────────────────────────────────────────────────────────

    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("Model")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        # Backend selector
        radio_row = QHBoxLayout()
        radio_row.addWidget(QLabel("Backend:"))
        self._radio_claude = QRadioButton("Claude (Anthropic)")
        self._radio_ollama = QRadioButton("Local (Ollama)")
        self._radio_claude.setChecked(True)
        bg = QButtonGroup(self)
        bg.addButton(self._radio_claude)
        bg.addButton(self._radio_ollama)
        self._radio_claude.toggled.connect(self._on_provider_changed)
        radio_row.addWidget(self._radio_claude)
        radio_row.addWidget(self._radio_ollama)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        # Stacked widget — one page per provider
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_claude_page())   # index 0
        self._stack.addWidget(self._build_ollama_page())   # index 1
        layout.addWidget(self._stack)

        return box

    def _build_claude_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("API Key:"))
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("sk-ant-… (or set ANTHROPIC_API_KEY)")
        self._key_input.textChanged.connect(self._validate)
        key_row.addWidget(self._key_input, stretch=1)
        self._btn_show_key = QPushButton("Show")
        self._btn_show_key.setFixedWidth(48)
        self._btn_show_key.setCheckable(True)
        self._btn_show_key.toggled.connect(self._toggle_key_vis)
        key_row.addWidget(self._btn_show_key)
        layout.addLayout(key_row)

        self._lbl_key_status = QLabel("")
        self._lbl_key_status.setStyleSheet("font-size: 10px; color: #aaa;")
        layout.addWidget(self._lbl_key_status)

        # Populate from env-var if present (session-only — not persisted)
        env_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if env_key:
            self._key_input.setPlaceholderText("Using ANTHROPIC_API_KEY from environment")
            self._lbl_key_status.setText("Key found in environment (ANTHROPIC_API_KEY)")
            self._lbl_key_status.setStyleSheet("font-size: 10px; color: #44ff88;")

        return page

    def _build_ollama_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._combo_ollama = QComboBox()
        self._combo_ollama.setEditable(True)
        self._combo_ollama.setPlaceholderText("e.g. llama3.2")
        self._combo_ollama.setInsertPolicy(QComboBox.InsertPolicy.InsertAtTop)
        self._combo_ollama.currentTextChanged.connect(self._validate)
        model_row.addWidget(self._combo_ollama, stretch=1)
        self._btn_detect = QPushButton("Detect")
        self._btn_detect.setFixedWidth(60)
        self._btn_detect.clicked.connect(self._detect_ollama)
        model_row.addWidget(self._btn_detect)
        layout.addLayout(model_row)

        self._lbl_ollama_status = QLabel(
            "Ollama will be started automatically when you click Run. "
            "Optionally click Detect to list local models."
        )
        self._lbl_ollama_status.setStyleSheet("font-size: 10px; color: #aaa;")
        self._lbl_ollama_status.setWordWrap(True)
        layout.addWidget(self._lbl_ollama_status)

        return page

    # ── Sandbox / Docker group ────────────────────────────────────────────────

    def _build_sandbox_group(self) -> QGroupBox:
        """
        Explains sandboxing options and lets the user configure max retries.

        Sandbox mode
        ------------
        By default the generated script is validated in a native subprocess
        against 1–2 sample images before running on the full dataset.

        Docker mode (BIOVISION_USE_DOCKER=1)
        -------------------------------------
        Set the environment variable BIOVISION_USE_DOCKER=1 before launching
        BioVision to run validation inside a locked-down Docker container
        (--network none, --memory 2g, read-only input mount).  Requires a
        pre-built image:
            docker build -t biovision-sandbox:latest .
        Docker is strongly recommended for untrusted scripts or production use.
        """
        box = QGroupBox("Sandbox & Retries")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        # Docker status row
        docker_row = QHBoxLayout()
        use_docker = os.environ.get("BIOVISION_USE_DOCKER", "0") == "1"
        docker_icon = "🐳 Docker" if use_docker else "🖥 Native subprocess"
        docker_color = "#44aaff" if use_docker else "#888"
        lbl_docker = QLabel(f"Sandbox mode: <b>{docker_icon}</b>")
        lbl_docker.setTextFormat(Qt.TextFormat.RichText)
        lbl_docker.setStyleSheet(f"font-size: 10px; color: {docker_color};")
        docker_row.addWidget(lbl_docker, stretch=1)
        layout.addLayout(docker_row)

        # Docker help text
        if not use_docker:
            lbl_hint = QLabel(
                "To enable Docker sandboxing: set <code>BIOVISION_USE_DOCKER=1</code> "
                "in your environment before launching, then build the sandbox image with "
                "<code>docker build -t biovision-sandbox:latest .</code>"
            )
            lbl_hint.setTextFormat(Qt.TextFormat.RichText)
            lbl_hint.setStyleSheet("font-size: 9px; color: #666;")
            lbl_hint.setWordWrap(True)
            layout.addWidget(lbl_hint)

        # Max retries row
        retry_row = QHBoxLayout()
        retry_row.addWidget(QLabel("Max retries:"))
        self._spin_retries = QSpinBox()
        self._spin_retries.setRange(1, 10)
        self._spin_retries.setValue(3)
        self._spin_retries.setFixedWidth(60)
        self._spin_retries.setToolTip(
            "Number of times the coder will attempt to fix a failing script "
            "before the pipeline aborts. Default: 3."
        )
        retry_row.addWidget(self._spin_retries)
        retry_row.addStretch()
        layout.addLayout(retry_row)

        return box

    # ── Activity indicator ────────────────────────────────────────────────────

    def _build_activity_widget(self) -> QFrame:
        """Compact pulsing-spinner area shown only while the pipeline is running."""
        self._activity_frame = QFrame()
        self._activity_frame.setStyleSheet(
            "QFrame { border: 1px solid #2a2a2a; border-radius: 4px; "
            "background: #1a1a1a; }"
        )
        self._activity_layout = QVBoxLayout(self._activity_frame)
        self._activity_layout.setContentsMargins(8, 4, 8, 4)
        self._activity_layout.setSpacing(2)
        self._activity_frame.setVisible(False)
        return self._activity_frame

    # ── Run bar ───────────────────────────────────────────────────────────────

    def _build_run_bar(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)

        self._lbl_run_status = QLabel("● Idle")
        self._lbl_run_status.setStyleSheet("font-size: 11px; color: #aaa;")
        layout.addWidget(self._lbl_run_status, stretch=1)

        self._btn_run = QPushButton("Run Agent")
        self._btn_run.clicked.connect(self._run_agent)
        layout.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_agent)
        layout.addWidget(self._btn_stop)

        return w

    # ── Resource monitor bar ──────────────────────────────────────────────────

    def _build_resource_bar(self) -> QWidget:
        """Compact CPU/RAM/GPU strip, hidden when pipeline is not running."""
        self._resource_bar = QWidget()
        layout = QHBoxLayout(self._resource_bar)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(12)

        self._resource_lbl = QLabel("CPU: — %   RAM: — MB")
        self._resource_lbl.setStyleSheet("font-size: 9px; color: #666; font-family: monospace;")
        layout.addWidget(self._resource_lbl)
        layout.addStretch()

        self._resource_bar.setVisible(False)
        return self._resource_bar

    # ── Log output ────────────────────────────────────────────────────────────

    def _build_log(self) -> QPlainTextEdit:
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(5000)
        self._output.setStyleSheet("font-family: monospace; font-size: 10px;")
        return self._output

    # ── Drag / drop ───────────────────────────────────────────────────────────

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drop_zone.set_hover(True)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self._drop_zone.set_hover(False)

    def dropEvent(self, event) -> None:
        self._drop_zone.set_hover(False)
        urls = event.mimeData().urls()
        accepted = False
        for url in urls:
            local = url.toLocalFile()
            if not local:
                continue
            p = Path(local)
            if p.is_dir():
                self._set_input_dir(str(p))
                accepted = True
            elif p.is_file() and p.suffix.lower() in {".yaml", ".yml"}:
                self._set_yaml_path(str(p))
                accepted = True
        if accepted:
            event.acceptProposedAction()
        else:
            self._log_err(
                "Drop rejected: only YAML files (.yaml/.yml) and folders are accepted."
            )
            event.ignore()

    # ── Browse buttons ────────────────────────────────────────────────────────

    def _browse_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select metadata YAML", self._yaml_path or "",
            "YAML Files (*.yaml *.yml)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if path:
            self._set_yaml_path(path)

    def _browse_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select image folder",
            self._input_dir or "",
            options=(
                QFileDialog.Option.ShowDirsOnly |
                QFileDialog.Option.DontUseNativeDialog
            ),
        )
        if path:
            self._set_input_dir(path)

    # ── Text field edit handlers ──────────────────────────────────────────────

    def _on_yaml_edited(self) -> None:
        text = self._edit_yaml.text().strip()
        if text and Path(text).is_file():
            self._set_yaml_path(text)
        elif text:
            self._lbl_input_status.setText("YAML path not found.")
            self._lbl_input_status.setStyleSheet("font-size: 10px; color: #ffaa44;")

    def _on_dir_edited(self) -> None:
        text = self._edit_dir.text().strip()
        if text and Path(text).is_dir():
            self._set_input_dir(text)
        elif text:
            self._lbl_input_status.setText("Folder not found.")
            self._lbl_input_status.setStyleSheet("font-size: 10px; color: #ffaa44;")

    # ── Input setters ─────────────────────────────────────────────────────────

    def _set_yaml_path(self, path: str) -> None:
        self._yaml_path = str(Path(path).resolve())
        self._edit_yaml.setText(self._yaml_path)
        self._edit_yaml.setToolTip(self._yaml_path)
        self._save_prefs_now()
        self._validate()

    def _set_input_dir(self, path: str) -> None:
        self._input_dir = str(Path(path).resolve())
        self._edit_dir.setText(self._input_dir)
        self._edit_dir.setToolTip(self._input_dir)
        # Derive output dir (same parent, suffix _biovision_output)
        p = Path(self._input_dir)
        self._output_dir = str(p.parent / (p.name + "_biovision_output"))
        self._lbl_out.setText(Path(self._output_dir).name)
        self._lbl_out.setToolTip(self._output_dir)
        self._save_prefs_now()
        self._validate()
        # Notify main widget so images can be auto-loaded in the viewer.
        self.input_dir_changed.emit(self._input_dir)

    # ── Provider toggle ───────────────────────────────────────────────────────

    def _on_provider_changed(self, checked: bool) -> None:
        if checked:  # Claude radio is now checked
            self._stack.setCurrentIndex(0)
        else:        # Ollama radio is now checked
            self._stack.setCurrentIndex(1)
            # Always refresh model list when switching to Ollama.
            self._detect_ollama()
        self._save_prefs_now()
        self._validate()

    # ── Ollama discovery ──────────────────────────────────────────────────────

    def _detect_ollama(self) -> None:
        if self._ollama_worker is not None:
            return  # already running
        self._btn_detect.setEnabled(False)
        self._lbl_ollama_status.setText("Detecting Ollama…")
        self._lbl_ollama_status.setStyleSheet("font-size: 10px; color: #888;")

        self._ollama_worker = _fetch_ollama_models_worker(_OLLAMA_DEFAULT_URL)
        self._ollama_worker.yielded.connect(self._on_ollama_models)
        self._ollama_worker.errored.connect(self._on_ollama_error)
        self._ollama_worker.finished.connect(
            lambda: (
                self._btn_detect.setEnabled(True),
                setattr(self, "_ollama_worker", None),
            )
        )
        self._ollama_worker.start()

    def _on_ollama_models(self, models: list) -> None:
        current = self._combo_ollama.currentText()
        self._combo_ollama.clear()
        if models:
            for m in models:
                self._combo_ollama.addItem(m)
            # Restore previous selection if still available.
            if current in models:
                self._combo_ollama.setCurrentText(current)
            else:
                self._combo_ollama.setCurrentIndex(0)
            self._lbl_ollama_status.setText(
                f"Ollama running — {len(models)} model(s) found."
            )
            self._lbl_ollama_status.setStyleSheet("font-size: 10px; color: #44ff88;")
        else:
            self._lbl_ollama_status.setText(
                "Ollama reachable but no models installed. Type a model name above."
            )
            self._lbl_ollama_status.setStyleSheet("font-size: 10px; color: #ffcc44;")
        self._validate()

    def _on_ollama_error(self, exc: Exception) -> None:
        self._lbl_ollama_status.setText(
            "Ollama not running — it will be started automatically when you click Run. "
            "Type a model name or leave blank to use the default (llama3.2)."
        )
        self._lbl_ollama_status.setStyleSheet("font-size: 10px; color: #aaa;")
        self._validate()

    # ── Activity indicator helpers ────────────────────────────────────────────

    def _start_activity(self, text: str) -> None:
        """Show the activity widget, clear any previous lines, start the spinner."""
        while self._activity_layout.count():
            item = self._activity_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._retry_count   = 0
        self._spinner_frame = 0
        self._activity_text = text

        lbl = QLabel()
        lbl.setStyleSheet(
            "border: none; font-size: 10px; color: #dddddd; font-family: monospace;"
        )
        self._activity_layout.addWidget(lbl)
        self._activity_current = lbl

        self._activity_frame.setVisible(True)
        self._spinner_timer.start(100)
        self._pulse_spinner()

    def _pulse_spinner(self) -> None:
        if self._activity_current is None:
            return
        frame = self._spinner_frames[self._spinner_frame % len(self._spinner_frames)]
        self._activity_current.setText(f"{frame}  {self._activity_text}")
        self._spinner_frame += 1

    def _update_activity(self, text: str) -> None:
        self._activity_text = text

    def _add_activity_line(self, text: str) -> None:
        if self._activity_current is not None:
            self._activity_current.setText(f"↺  {self._activity_text}")
            self._activity_current.setStyleSheet(
                "border: none; font-size: 10px; color: #555; font-family: monospace;"
            )

        self._activity_text = text
        lbl = QLabel()
        lbl.setStyleSheet(
            "border: none; font-size: 10px; color: #dddddd; font-family: monospace;"
        )
        self._activity_layout.addWidget(lbl)
        self._activity_current = lbl
        self._pulse_spinner()

    def _stop_activity(self) -> None:
        self._spinner_timer.stop()
        if self._activity_frame is not None:
            self._activity_frame.setVisible(False)
        if self._activity_layout is not None:
            while self._activity_layout.count():
                item = self._activity_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        self._activity_current = None

    def _advance_activity(self, node_name: str, state_delta: dict) -> None:
        if node_name == "_preflight":
            msg = state_delta.get("message", "")
            if msg:
                self._update_activity(f"Preparing Ollama — {msg}")
            return

        if node_name == "planner":
            self._update_activity("Generating Python script…")

        elif node_name == "coder":
            if self._retry_count > 0:
                self._update_activity(
                    f"Re-validating corrected script (attempt {self._retry_count + 1})…"
                )
            else:
                self._update_activity("Validating script on sample images…")

        elif node_name == "sandbox_executor":
            if state_delta.get("execution_success"):
                self._update_activity("Running script on full dataset…")
            else:
                self._retry_count += 1
                self._add_activity_line(
                    f"Retry {self._retry_count} — Regenerating script with error corrections…"
                )

        elif node_name in ("local_executor", "terminal_failure"):
            self._stop_activity()

    # ── Resource monitor ──────────────────────────────────────────────────────

    def _update_resources(self) -> None:
        r = _read_resources()
        parts = [f"CPU: {r['cpu']:.0f}%", f"RAM: {r['ram_mb']} MB"]
        if r["gpu_pct"] is not None:
            parts.append(f"GPU: {r['gpu_pct']}%  VRAM: {r['gpu_mem_mb']} MB")
        self._resource_lbl.setText("   ".join(parts))

    # ── Key visibility toggle ─────────────────────────────────────────────────

    def _toggle_key_vis(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self._key_input.setEchoMode(mode)
        self._btn_show_key.setText("Hide" if visible else "Show")

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        """Update status labels and enable/disable the Run button."""
        issues = []

        if not self._yaml_path or not Path(self._yaml_path).is_file():
            issues.append("No metadata YAML selected.")
        if not self._input_dir or not Path(self._input_dir).is_dir():
            issues.append("No image folder selected.")
        else:
            n = _count_images(self._input_dir)
            if n == 0:
                issues.append("Image folder contains no supported image files.")
            else:
                self._lbl_input_status.setText(
                    f"{n} supported image file(s) found in folder."
                )
                self._lbl_input_status.setStyleSheet("font-size: 10px; color: #44ff88;")

        if issues:
            self._lbl_input_status.setText("  ".join(issues))
            self._lbl_input_status.setStyleSheet("font-size: 10px; color: #ffaa44;")

        provider = self._current_provider()
        if provider == "anthropic":
            key_in_env = bool(os.environ.get("ANTHROPIC_API_KEY", ""))
            key_typed  = bool(self._key_input.text().strip())
            if not key_in_env and not key_typed:
                issues.append("Anthropic API key required.")
        elif provider == "ollama":
            if not self._combo_ollama.currentText().strip():
                self._lbl_ollama_status.setText(
                    "No model entered — default model 'llama3.2' will be used."
                )
                self._lbl_ollama_status.setStyleSheet("font-size: 10px; color: #aaa;")

        can_run = not issues and self._worker is None
        self._btn_run.setEnabled(can_run)

        if not issues:
            self._lbl_run_status.setText("● Ready")
            self._lbl_run_status.setStyleSheet("font-size: 11px; color: #44ff88;")
        elif self._worker is None:
            self._lbl_run_status.setText("● Not ready — " + issues[0])
            self._lbl_run_status.setStyleSheet("font-size: 11px; color: #ffaa44;")

    # ── Run / stop ────────────────────────────────────────────────────────────

    def _run_agent(self) -> None:
        self._output.clear()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._lbl_run_status.setText("● Running…")
        self._lbl_run_status.setStyleSheet("font-size: 11px; color: #ffcc44;")

        provider   = self._current_provider()
        model      = self._current_model()
        api_key    = self._current_api_key()
        base_url   = _OLLAMA_DEFAULT_URL if provider == "ollama" else ""
        max_retries = self._spin_retries.value()

        try:
            with open(self._yaml_path, encoding="utf-8") as fh:
                metadata_yaml = fh.read()
        except Exception as exc:
            self._log_err(f"Could not read YAML file: {exc}")
            self._on_agent_error(exc)
            return

        _model_display = model or ("(default: llama3.2)" if provider == "ollama" else "(default)")
        self._log(f"[{_ts()}] Provider: {provider}  Model: {_model_display}  Max retries: {max_retries}")
        self._log(f"[{_ts()}] Input:    {self._input_dir}")
        self._log(f"[{_ts()}] Output:   {self._output_dir}")
        self._log("")

        _initial = "Preparing Ollama…" if provider == "ollama" else "Planning preprocessing steps…"
        self._start_activity(_initial)

        # Start resource monitor
        self._resource_bar.setVisible(True)
        self._resource_timer.start(2000)
        self._update_resources()

        # Notify LLM chat widget to sync connection.
        self.llm_config_ready.emit(provider, model, api_key, base_url)

        self._worker = run_biovision_agent_worker(
            metadata_yaml=metadata_yaml,
            input_dir=self._input_dir,
            output_dir=self._output_dir,
            llm_provider=provider,
            llm_model=model,
            llm_api_key=api_key,
            llm_base_url=base_url,
            max_retries=max_retries,
        )
        self._worker.yielded.connect(self._on_node_done)
        self._worker.finished.connect(self._on_agent_finished)
        self._worker.errored.connect(self._on_agent_error)
        self._worker.start()

    def _stop_agent(self) -> None:
        if self._worker is not None:
            self._worker.quit()
            self._log(f"[{_ts()}] Stop requested — worker will finish current operation.")

    # ── Worker callbacks ──────────────────────────────────────────────────────

    def _on_node_done(self, payload: tuple) -> None:
        node_name, state_delta = payload

        if node_name == "_preflight":
            msg = state_delta.get("message", "")
            if msg:
                self._log(f"[{_ts()}] [Preflight] {msg}")
            self._advance_activity(node_name, state_delta)
            return

        label = _node_label(node_name)
        self._lbl_run_status.setText(f"● {label}")
        self._log(f"[{_ts()}] [{node_name}] {label}")

        if node_name == "planner":
            title = state_delta.get("plan_title", "")
            steps = state_delta.get("plan_steps", [])
            rationale = state_delta.get("plan_rationale", "")
            error = state_delta.get("error")
            if error:
                self._log_err(f"  ERROR: {error}")
            else:
                if title:
                    self._log(f"  Plan: {title}")
                if rationale:
                    self._log(f"  Rationale: {rationale}")
                if steps:
                    self._log(f"  Steps ({len(steps)}):")
                    for i, s in enumerate(steps, 1):
                        self._log(f"    {i}. {s}")

        elif node_name == "coder":
            deps = state_delta.get("code_dependencies", [])
            code_len = len(state_delta.get("generated_code", ""))
            error = state_delta.get("error")
            if error:
                self._log_err(f"  ERROR: {error}")
            else:
                if code_len:
                    self._log(f"  Script: {code_len} chars")
                if deps:
                    self._log(f"  Dependencies: {', '.join(deps)}")

        elif node_name in ("sandbox_executor", "local_executor"):
            success = state_delta.get("execution_success")
            stdout  = state_delta.get("execution_stdout", "")
            stderr  = state_delta.get("execution_stderr", "")
            retries = state_delta.get("retries", 0)
            status  = "✓ OK" if success else "✗ FAILED"
            self._log(f"  Status: {status}")
            if not success and retries:
                max_r = self._spin_retries.value()
                self._log(f"  Retry {retries}/{max_r}")
            if stdout:
                self._log(f"  stdout:\n{stdout}")
            if stderr:
                if not success:
                    self._log_err(f"  stderr:\n{stderr}")
                else:
                    self._log(f"  stderr:\n{stderr}")

        elif node_name == "terminal_failure":
            error = state_delta.get("error", "Max retries exhausted.")
            self._log_err(f"  TERMINAL FAILURE: {error}")

        self._advance_activity(node_name, state_delta)

    def _on_agent_finished(self) -> None:
        self._stop_activity()
        self._resource_timer.stop()
        self._resource_bar.setVisible(False)
        self._lbl_run_status.setText("● Done")
        self._lbl_run_status.setStyleSheet("font-size: 11px; color: #44ff88;")
        self._log(f"\n[{_ts()}] Pipeline finished.")
        self._worker = None
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._validate()
        self.agent_finished.emit()

    def _on_agent_error(self, exc: Exception) -> None:
        self._stop_activity()
        self._resource_timer.stop()
        self._resource_bar.setVisible(False)
        msg = str(exc)
        self._lbl_run_status.setText("● Error")
        self._lbl_run_status.setStyleSheet("font-size: 11px; color: #ff4444;")
        self._log_err(f"\n[{_ts()}] ERROR: {msg}")
        self._worker = None
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._validate()

    # ── Accessors ─────────────────────────────────────────────────────────────

    def _current_provider(self) -> str:
        return "anthropic" if self._radio_claude.isChecked() else "ollama"

    def _current_model(self) -> str:
        if self._radio_claude.isChecked():
            return ""
        return self._combo_ollama.currentText().strip()

    def _current_api_key(self) -> str:
        """Return the Anthropic key — typed field takes priority over env-var.
        Never logged or written to disk."""
        typed = self._key_input.text().strip()
        return typed or os.environ.get("ANTHROPIC_API_KEY", "")

    # ── Preferences (non-secret only) ─────────────────────────────────────────

    def _load_prefs(self) -> None:
        prefs = _load_prefs()
        yaml_path = prefs.get("yaml_path", "")
        input_dir = prefs.get("input_dir", "")
        provider  = prefs.get("provider", "anthropic")
        model     = prefs.get("ollama_model", "")
        retries   = prefs.get("max_retries", 3)

        if yaml_path and Path(yaml_path).is_file():
            self._set_yaml_path(yaml_path)
        if input_dir and Path(input_dir).is_dir():
            self._set_input_dir(input_dir)

        if provider == "ollama":
            self._radio_ollama.setChecked(True)
        else:
            self._radio_claude.setChecked(True)

        if model:
            self._combo_ollama.setCurrentText(model)

        self._spin_retries.setValue(max(1, min(10, int(retries))))

    def _save_prefs_now(self) -> None:
        prefs = {
            "yaml_path":    self._yaml_path,
            "input_dir":    self._input_dir,
            "provider":     self._current_provider(),
            "ollama_model": self._combo_ollama.currentText().strip(),
            "max_retries":  self._spin_retries.value(),
        }
        _save_prefs(prefs)

    # ── Log helpers ───────────────────────────────────────────────────────────

    def _log(self, text: str) -> None:
        self._output.appendPlainText(text)
        self._output.moveCursor(QTextCursor.MoveOperation.End)

    def _log_err(self, text: str) -> None:
        # Escape HTML special chars so angle brackets in tracebacks render correctly.
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._output.appendHtml(
            f'<span style="color:#ff6666; font-family:monospace;">{safe}</span>'
        )
        self._output.moveCursor(QTextCursor.MoveOperation.End)
