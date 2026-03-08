"""
AgentChatPanel — unified BioVision agent + interactive chat panel.

Layout
------
  ┌─ QTabWidget ──────────────────────────────────────────────┐
  │  [Directories]  [Configure]                               │
  │                                                           │
  │  Directories tab:                                         │
  │    YAML:   [editable path]  [Browse]                      │
  │    Folder: [editable path]  [Browse]                      │
  │    Output: derived path                                   │
  │    [Drop zone]                                            │
  │                                                           │
  │  Configure tab:                                           │
  │    [● Connected: Anthropic/claude]  [Change ▾]            │
  │    [Setup section — collapses after connect]              │
  │    ──────────────────────────────────────────             │
  │    Max retries: [3]  [🖥 Native]   [Run Agent]  [Stop]    │
  └───────────────────────────────────────────────────────────┘
  ⠋  Planning…                              CPU 12%  RAM 342MB
  ──────────────────────────────────────────────────────────────
  [Chat display — pipeline events + conversation]
  ...
  ──────────────────────────────────────────────────────────────
  [Message input…]                            [Send]  [Clear]

Signals
-------
  agent_finished      — pipeline completed (success or failure)
  input_dir_changed   — image folder changed (main widget loads images)
  config_patch_applied — LLM suggested and applied a yaml-patch
"""
from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from napari.qt.threading import thread_worker

from biovision_napari.state.project_state import ProjectState
# Reuse connection primitives already in llm_chat.py
from biovision_napari.ui.llm_chat import (
    _Connection,
    _DOT_GREEN,
    _DOT_RED,
    _DOT_YELLOW,
    _SetupPanel,
    _StatusBanner,
    _detect_connection,
    _llm_call_worker,
    extract_patches,
)
from biovision_napari.workers.agent_worker import run_biovision_agent_worker

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".ndpi", ".svs"}
_OLLAMA_DEFAULT_URL = "http://localhost:11434"


def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _count_images(folder: str) -> int:
    try:
        return sum(
            1 for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
    except Exception:
        return 0


def _read_resources() -> dict:
    r: dict = {"cpu": None, "ram_mb": None, "gpu_pct": None, "gpu_mem_mb": None}
    try:
        import psutil
        r["cpu"] = psutil.cpu_percent(interval=None)
        r["ram_mb"] = int(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()[0]
        parts = out.split(",")
        r["gpu_pct"] = int(parts[0].strip())
        r["gpu_mem_mb"] = int(parts[1].strip())
    except Exception:
        pass
    return r


_NODE_LABELS = {
    "planner":          "Planning preprocessing steps",
    "coder":            "Generating Python script",
    "sandbox_executor": "Validating on sample images",
    "local_executor":   "Running on full dataset",
    "terminal_failure": "Pipeline failed — max retries exhausted",
}

_SYSTEM_PROMPT = """\
You are the BioVision assistant — an expert in bioimage analysis, microscopy
preprocessing, and the BioVision napari plugin.

You can help the user:
- Configure and run the preprocessing pipeline.
- Understand and adjust viewer.yaml settings.
- Interpret pipeline results and suggest improvements.
- Troubleshoot errors in the generated scripts.

When you propose a change to viewer.yaml, output it as a YAML block tagged:
```yaml-patch
<only the keys to update>
```

Be concise, practical, and specific to bioimage analysis.
"""

_DOCKER_GREETING = """\
Hi! I'm the BioVision agent. I can run your image preprocessing pipeline and \
help you configure your segmentation workflow.

**Docker sandboxing** (recommended for production):

**Step 1** — Build the sandbox image once, from the repo root:
```
docker build -t biovision-sandbox:latest .
```

**Step 2** — Enable Docker mode, then relaunch BioVision:

macOS / Linux (bash/zsh):
```
export BIOVISION_USE_DOCKER=1
```

Windows PowerShell:
```
$env:BIOVISION_USE_DOCKER = "1"
```

Windows Command Prompt:
```
set BIOVISION_USE_DOCKER=1
```

Without Docker, validation runs in a native subprocess (safe for development). \
Set your paths in the **Directories** tab, configure the LLM in the **Configure** tab, \
then click **Run Agent** — or ask me anything!\
"""

_PREFS_PATH = Path.home() / ".biovision" / "agent_prefs.json"


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


# ---------------------------------------------------------------------------
# Pipeline graph visualiser
# ---------------------------------------------------------------------------

class _PipelineGraphWidget(QWidget):
    """
    Compact horizontal strip showing pipeline node states in real-time.

    [Plan] → [Code] → [Sandbox] → [Run] → Done
    """

    _NODES  = ["planner", "coder", "sandbox_executor", "local_executor"]
    _LABELS = {
        "planner":          "Plan",
        "coder":            "Code",
        "sandbox_executor": "Sandbox",
        "local_executor":   "Full Run",
    }
    _B = "border-radius:3px;font-size:9px;padding:2px 6px;min-width:52px;"
    _S = {
        "idle":   _B + "background:#1e1e1e;color:#444;border:1px solid #333;",
        "active": _B + "background:#0d2845;color:#44aaff;border:1px solid #44aaff;font-weight:bold;",
        "ok":     _B + "background:#0d2d1a;color:#44ff88;border:1px solid #44ff88;",
        "retry":  _B + "background:#2d1e00;color:#ffaa44;border:1px solid #ffaa44;",
        "fail":   _B + "background:#2d0d0d;color:#ff6666;border:1px solid #ff6666;",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lbl: dict[str, QLabel] = {}
        self._end_lbl: Optional[QLabel] = None
        self._abort_lbl: Optional[QLabel] = None
        self._build()
        self.reset()

    def _build(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(3)

        title = QLabel("Pipeline:")
        title.setStyleSheet("color:#444;font-size:9px;")
        lay.addWidget(title)

        for i, nid in enumerate(self._NODES):
            if i:
                arr = QLabel("→")
                arr.setStyleSheet("color:#2a2a2a;font-size:9px;")
                lay.addWidget(arr)
            lbl = QLabel(self._LABELS[nid])
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._lbl[nid] = lbl
            lay.addWidget(lbl)

        arr = QLabel("→")
        arr.setStyleSheet("color:#2a2a2a;font-size:9px;")
        lay.addWidget(arr)

        self._end_lbl = QLabel("Done")
        self._end_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._end_lbl)

        self._abort_lbl = QLabel("✗ Aborted")
        self._abort_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._abort_lbl.setVisible(False)
        lay.addWidget(self._abort_lbl)

        lay.addStretch()

    def reset(self):
        for nid, lbl in self._lbl.items():
            lbl.setText(self._LABELS[nid])
            lbl.setStyleSheet(self._S["idle"])
        if self._end_lbl:
            self._end_lbl.setText("Done")
            self._end_lbl.setStyleSheet(self._S["idle"])
        if self._abort_lbl:
            self._abort_lbl.setVisible(False)

    def set_active(self, node: str):
        if node in self._lbl:
            self._lbl[node].setText(self._LABELS[node] + " ●")
            self._lbl[node].setStyleSheet(self._S["active"])

    def set_done(self, node: str, retries: int = 0):
        if node in self._lbl:
            suffix = " ✓" if not retries else f" ✓(↺{retries})"
            self._lbl[node].setText(self._LABELS[node] + suffix)
            self._lbl[node].setStyleSheet(self._S["ok"])

    def set_retry(self, node: str, n: int):
        if node in self._lbl:
            self._lbl[node].setText(f"{self._LABELS[node]} ↺{n}")
            self._lbl[node].setStyleSheet(self._S["retry"])

    def set_failed(self, node: str):
        if node in self._lbl:
            self._lbl[node].setText(self._LABELS[node] + " ✗")
            self._lbl[node].setStyleSheet(self._S["fail"])

    def set_terminal_failure(self):
        if self._abort_lbl:
            self._abort_lbl.setVisible(True)
            self._abort_lbl.setStyleSheet(self._S["fail"])

    def set_pipeline_done(self):
        if self._end_lbl:
            self._end_lbl.setText("Done ✓")
            self._end_lbl.setStyleSheet(self._S["ok"])


# ---------------------------------------------------------------------------
# Drop-zone
# ---------------------------------------------------------------------------

class _DropZone(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Drop YAML or image folder here")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._ns = "border:2px dashed #444;border-radius:4px;color:#666;font-size:9px;padding:4px;"
        self._hs = "border:2px dashed #44aaff;border-radius:4px;color:#44aaff;font-size:9px;padding:4px;"
        self.setStyleSheet(self._ns)

    def set_hover(self, active: bool) -> None:
        self.setStyleSheet(self._hs if active else self._ns)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class AgentChatPanel(QWidget):
    """
    Unified agent + chat panel.  A single LLM connection powers both the
    preprocessing pipeline and the conversational assistant.
    """

    agent_finished      = Signal()
    input_dir_changed   = Signal(str)
    config_patch_applied = Signal(str)

    def __init__(self, state: ProjectState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state

        self._conn:   Optional[_Connection] = None
        self._history: list[dict] = []

        # Workers (one pipeline, one chat — never concurrent with each other).
        self._pipeline_worker = None
        self._chat_worker     = None

        # LLM token streaming state
        self._stream_active = False
        self._stream_node   = ""

        # Spinner animation
        self._spinner_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._spinner_idx    = 0
        self._spinner_timer  = QTimer(self)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_lbl: Optional[QLabel] = None

        # Resource monitor
        self._resource_timer = QTimer(self)
        self._resource_timer.timeout.connect(self._update_resources)

        # Paths
        self._yaml_path  = ""
        self._input_dir  = ""
        self._output_dir = ""

        self.setAcceptDrops(True)
        self._build_ui()

        self._state.config_changed.connect(self._on_config_changed)

        # Restore saved paths/prefs.
        self._load_prefs()
        self._validate_inputs()

        # Auto-detect any already-available LLM connection.
        QTimer.singleShot(400, self._auto_connect)

    # ── Build UI ─────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # 1. Sub-tabs: Directories | Configure
        self._sub_tabs = QTabWidget()
        self._sub_tabs.addTab(self._build_directories_tab(), "Directories")
        self._sub_tabs.addTab(self._build_configure_tab(), "Configure")
        outer.addWidget(self._sub_tabs)

        # 2. Pipeline graph visualiser (always visible)
        self._graph = _PipelineGraphWidget()
        outer.addWidget(self._graph)

        # 3. Spinner row (hidden when idle)
        self._spinner_row = QWidget()
        sr_layout = QHBoxLayout(self._spinner_row)
        sr_layout.setContentsMargins(4, 0, 4, 0)
        self._spinner_lbl = QLabel("")
        self._spinner_lbl.setStyleSheet("font-size:10px;color:#aaa;font-family:monospace;")
        sr_layout.addWidget(self._spinner_lbl)
        self._resource_lbl = QLabel("")
        self._resource_lbl.setStyleSheet("font-size:9px;color:#555;font-family:monospace;")
        sr_layout.addStretch()
        sr_layout.addWidget(self._resource_lbl)
        self._spinner_row.setVisible(False)
        outer.addWidget(self._spinner_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#333;")
        outer.addWidget(sep)

        # 3. Chat display
        self._display = QPlainTextEdit()
        self._display.setReadOnly(True)
        self._display.setMaximumBlockCount(6000)
        self._display.setStyleSheet("font-size:11px; font-family:monospace;")
        outer.addWidget(self._display, stretch=1)

        # 4. Input row
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask BioVision assistant…")
        self._input.returnPressed.connect(self._send_chat)
        self._input.setEnabled(False)
        input_row.addWidget(self._input, stretch=1)

        self._btn_send = QPushButton("Send")
        self._btn_send.clicked.connect(self._send_chat)
        self._btn_send.setEnabled(False)
        input_row.addWidget(self._btn_send)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_chat)
        input_row.addWidget(btn_clear)

        outer.addLayout(input_row)

    def _build_directories_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # YAML row
        yaml_row = QHBoxLayout()
        yaml_row.addWidget(QLabel("YAML:"))
        self._edit_yaml = QLineEdit()
        self._edit_yaml.setPlaceholderText("metadata.yaml path…")
        self._edit_yaml.setStyleSheet("font-size:10px;")
        self._edit_yaml.editingFinished.connect(self._on_yaml_edited)
        yaml_row.addWidget(self._edit_yaml, stretch=1)
        btn_yaml = QPushButton("Browse…")
        btn_yaml.setFixedWidth(68)
        btn_yaml.clicked.connect(self._browse_yaml)
        yaml_row.addWidget(btn_yaml)
        layout.addLayout(yaml_row)

        # Image folder row
        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("Folder:"))
        self._edit_dir = QLineEdit()
        self._edit_dir.setPlaceholderText("Image folder path…")
        self._edit_dir.setStyleSheet("font-size:10px;")
        self._edit_dir.editingFinished.connect(self._on_dir_edited)
        dir_row.addWidget(self._edit_dir, stretch=1)
        btn_dir = QPushButton("Browse…")
        btn_dir.setFixedWidth(68)
        btn_dir.clicked.connect(self._browse_dir)
        dir_row.addWidget(btn_dir)
        layout.addLayout(dir_row)

        # Output label
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._lbl_out = QLabel("—")
        self._lbl_out.setStyleSheet("color:#555;font-size:10px;")
        out_row.addWidget(self._lbl_out, stretch=1)
        layout.addLayout(out_row)

        # Drop zone (compact)
        self._drop_zone = _DropZone()
        self._drop_zone.setFixedHeight(30)
        layout.addWidget(self._drop_zone)

        layout.addStretch()
        return w

    def _build_configure_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)

        # Status banner + collapsible setup panel
        self._banner = _StatusBanner()
        self._banner.setup_toggled.connect(self._on_setup_toggled)
        layout.addWidget(self._banner)

        self._setup = _SetupPanel()
        self._setup.connected.connect(self._on_connected)
        self._setup.setVisible(False)
        layout.addWidget(self._setup)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color:#333;")
        layout.addWidget(sep)

        # Run controls
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Max retries:"))
        self._spin_retries = QSpinBox()
        self._spin_retries.setRange(1, 10)
        self._spin_retries.setValue(3)
        self._spin_retries.setFixedWidth(52)
        ctrl_row.addWidget(self._spin_retries)

        use_docker = os.environ.get("BIOVISION_USE_DOCKER", "0") == "1"
        sandbox_lbl = QLabel("🐳 Docker" if use_docker else "🖥 Native")
        sandbox_lbl.setStyleSheet(
            f"font-size:9px;color:{'#44aaff' if use_docker else '#666'};"
        )
        sandbox_lbl.setToolTip(
            "Docker mode active (BIOVISION_USE_DOCKER=1)" if use_docker
            else "Native subprocess mode. See chat for Docker setup instructions."
        )
        ctrl_row.addWidget(sandbox_lbl)
        ctrl_row.addStretch()

        self._lbl_input_status = QLabel("")
        self._lbl_input_status.setStyleSheet("font-size:9px;color:#666;")
        ctrl_row.addWidget(self._lbl_input_status)

        self._btn_run = QPushButton("Run Agent")
        self._btn_run.clicked.connect(self._run_pipeline)
        self._btn_run.setEnabled(False)
        ctrl_row.addWidget(self._btn_run)

        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_pipeline)
        ctrl_row.addWidget(self._btn_stop)

        layout.addLayout(ctrl_row)
        layout.addStretch()
        return w

    # ── Connection management ─────────────────────────────────────────────────

    def _auto_connect(self) -> None:
        cfg = self._state.config
        conn = _detect_connection(cfg)
        if conn:
            self._on_connected(conn)
        else:
            self._banner.set_status(_DOT_RED, "No LLM connected — click Set up to configure")
            self._setup.setVisible(True)
            self._banner._btn_setup.setChecked(True)

    def _on_connected(self, conn: _Connection) -> None:
        self._conn = conn
        self._banner.set_status(_DOT_GREEN, f"Connected: {conn.label}")
        self._setup.setVisible(False)
        self._banner._btn_setup.setChecked(False)
        self._input.setEnabled(True)
        self._btn_send.setEnabled(True)
        self._validate_inputs()

        if not self._history:
            self._show_greeting()

    def _on_setup_toggled(self, checked: bool) -> None:
        self._setup.setVisible(checked)

    def _on_config_changed(self, config) -> None:
        if config and self._conn is None:
            self._auto_connect()

    # ── Greeting ──────────────────────────────────────────────────────────────

    def _show_greeting(self) -> None:
        self._append_assistant(_DOCKER_GREETING)
        self._history.append({"role": "assistant", "content": _DOCKER_GREETING})

    # ── Browse / drop ─────────────────────────────────────────────────────────

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
        for url in event.mimeData().urls():
            local = url.toLocalFile()
            if not local:
                continue
            p = Path(local)
            if p.is_dir():
                self._set_input_dir(str(p))
                event.acceptProposedAction()
            elif p.is_file() and p.suffix.lower() in {".yaml", ".yml"}:
                self._set_yaml_path(str(p))
                event.acceptProposedAction()

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
            self, "Select image folder", self._input_dir or "",
            options=(
                QFileDialog.Option.ShowDirsOnly |
                QFileDialog.Option.DontUseNativeDialog
            ),
        )
        if path:
            self._set_input_dir(path)

    def _on_yaml_edited(self) -> None:
        t = self._edit_yaml.text().strip()
        if t and Path(t).is_file():
            self._set_yaml_path(t)

    def _on_dir_edited(self) -> None:
        t = self._edit_dir.text().strip()
        if t and Path(t).is_dir():
            self._set_input_dir(t)

    def _set_yaml_path(self, path: str) -> None:
        self._yaml_path = str(Path(path).resolve())
        self._edit_yaml.setText(self._yaml_path)
        self._save_prefs_now()
        self._validate_inputs()

    def _set_input_dir(self, path: str) -> None:
        self._input_dir = str(Path(path).resolve())
        self._edit_dir.setText(self._input_dir)
        p = Path(self._input_dir)
        self._output_dir = str(p.parent / (p.name + "_biovision_output"))
        self._lbl_out.setText(Path(self._output_dir).name)
        self._lbl_out.setToolTip(self._output_dir)
        self._save_prefs_now()
        self._validate_inputs()
        self.input_dir_changed.emit(self._input_dir)

    # ── Input validation ──────────────────────────────────────────────────────

    def _validate_inputs(self) -> None:
        ok_yaml   = bool(self._yaml_path and Path(self._yaml_path).is_file())
        ok_dir    = bool(self._input_dir and Path(self._input_dir).is_dir())
        connected = self._conn is not None

        if ok_dir:
            n = _count_images(self._input_dir)
            self._lbl_input_status.setText(f"{n} image(s)" if n else "no images")
            self._lbl_input_status.setStyleSheet(
                f"font-size:9px;color:{'#44ff88' if n else '#ff6666'};"
            )
        else:
            self._lbl_input_status.setText("")

        can_run = ok_yaml and ok_dir and connected and self._pipeline_worker is None
        self._btn_run.setEnabled(can_run)

    # ── Pipeline run ──────────────────────────────────────────────────────────

    # ── Pre-run validation ────────────────────────────────────────────────────

    def _validate_before_run(self) -> Optional[str]:
        """Return a human-readable error string if inputs are not ready, else None."""
        # YAML path
        if not self._yaml_path:
            return (
                "No metadata YAML selected.\n"
                "→ Open the Directories tab and set a YAML path."
            )
        yaml_path = Path(self._yaml_path)
        if not yaml_path.is_file():
            return (
                f"YAML file not found:\n  {self._yaml_path}\n"
                "→ Check the path in the Directories tab."
            )
        try:
            with open(yaml_path, encoding="utf-8") as fh:
                meta = yaml.safe_load(fh.read())
        except Exception as exc:
            return (
                f"YAML parse error:\n  {exc}\n"
                "→ Fix the YAML syntax before running."
            )
        if not isinstance(meta, dict) or not meta:
            return (
                "The YAML file appears empty or contains no mappings.\n"
                "→ Add dataset metadata (instrument, modality, image format, etc.)."
            )

        # Image folder
        if not self._input_dir:
            return (
                "No image folder selected.\n"
                "→ Open the Directories tab and set an input folder."
            )
        input_path = Path(self._input_dir)
        if not input_path.is_dir():
            return (
                f"Image folder not found:\n  {self._input_dir}\n"
                "→ Check the path in the Directories tab."
            )
        n = _count_images(self._input_dir)
        if n == 0:
            exts = ", ".join(sorted(_IMAGE_EXTENSIONS))
            return (
                f"No supported images found in:\n  {self._input_dir}\n"
                f"→ Supported formats: {exts}"
            )
        return None

    def _run_pipeline(self) -> None:
        assert self._conn is not None

        err = self._validate_before_run()
        if err:
            self._append_system(f"⚠ Cannot run pipeline:\n{err}", error=True)
            self._sub_tabs.setCurrentIndex(0)  # jump to Directories tab
            return

        try:
            with open(self._yaml_path, encoding="utf-8") as fh:
                metadata_yaml = fh.read()
        except Exception as exc:
            self._append_system(f"ERROR reading YAML: {exc}", error=True)
            return

        # Resolve provider/model/key from the active connection.
        conn = self._conn
        if conn.provider == "anthropic":
            llm_provider = "anthropic"
            llm_model    = conn.model
            llm_api_key  = conn.api_key
            llm_base_url = ""
        else:
            # openai-compat (Groq, OpenAI, Ollama)
            llm_provider = "ollama" if "11434" in (conn.base_url or "") else "anthropic"
            llm_model    = conn.model
            llm_api_key  = conn.api_key if conn.api_key != "ollama" else ""
            llm_base_url = conn.base_url or ""

        max_retries = self._spin_retries.value()

        self._append_system(
            f"[{_ts()}] Starting pipeline — {conn.label} | retries: {max_retries}\n"
            f"  Input:  {self._input_dir}\n"
            f"  Output: {self._output_dir}"
        )

        # Switch to Configure tab to show run status
        self._sub_tabs.setCurrentIndex(1)

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._spinner_row.setVisible(True)
        self._spinner_timer.start(100)
        self._resource_timer.start(2000)
        self._update_resources()

        # Reset and activate the graph visualiser
        self._graph.reset()
        self._graph.set_active("planner")
        self._set_spinner("Analysing dataset metadata and building preprocessing plan…")

        self._pipeline_worker = run_biovision_agent_worker(
            metadata_yaml=metadata_yaml,
            input_dir=self._input_dir,
            output_dir=self._output_dir,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            max_retries=max_retries,
        )
        self._pipeline_worker.yielded.connect(self._on_pipeline_event)
        self._pipeline_worker.finished.connect(self._on_pipeline_done)
        self._pipeline_worker.errored.connect(self._on_pipeline_error)
        self._pipeline_worker.start()

    def _stop_pipeline(self) -> None:
        if self._pipeline_worker is not None:
            self._pipeline_worker.quit()
            self._append_system(f"[{_ts()}] Stop requested.")

    # ── LLM token streaming ───────────────────────────────────────────────────

    def _append_stream_token(self, node: str, token: str) -> None:
        """Append a single LLM output token to the chat display."""
        if not self._stream_active:
            node_label = _NODE_LABELS.get(node, node)
            safe = node_label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            self._display.appendHtml(
                f'<span style="color:#8888aa;font-size:10px;font-family:monospace;">'
                f'[{_ts()}] ⟳ {safe}:</span>'
            )
            self._stream_active = True
            self._stream_node   = node
        cursor = self._display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self._display.setTextCursor(cursor)
        self._display.ensureCursorVisible()

    def _finalize_stream(self) -> None:
        """Close any open token stream — insert a trailing newline."""
        if not self._stream_active:
            return
        cursor = self._display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n")
        self._display.setTextCursor(cursor)
        self._stream_active = False
        self._stream_node   = ""

    # ── Pipeline events ───────────────────────────────────────────────────────

    def _on_pipeline_event(self, payload: tuple) -> None:
        node_name, state_delta = payload

        # Token-level streaming — handle silently without any other processing.
        if node_name == "_token":
            self._append_stream_token(
                state_delta.get("node", ""),
                state_delta.get("token", ""),
            )
            return

        # Finalize any active token stream before printing node-completion message.
        self._finalize_stream()

        if node_name == "_preflight":
            msg = state_delta.get("message", "")
            if not msg:
                return
            _success_words = (
                "already running", "ready", "started successfully",
                "available locally", "pulled successfully",
            )
            icon = "✓" if any(w in msg.lower() for w in _success_words) else "⟳"
            self._set_spinner(msg)
            self._append_system(f"[{_ts()}] {icon} {msg}")
            return

        if node_name == "planner":
            title     = state_delta.get("plan_title", "")
            steps     = state_delta.get("plan_steps", [])
            rationale = state_delta.get("plan_rationale", "")
            err       = state_delta.get("error")
            if err:
                self._graph.set_failed("planner")
                self._append_system(f"[{_ts()}] ✗ Planner error: {err}", error=True)
            else:
                self._graph.set_done("planner")
                self._graph.set_active("coder")
                lines = [f"[{_ts()}] ✓ Plan ready: {title}"]
                if rationale:
                    lines.append(f"  Rationale: {rationale}")
                for i, s in enumerate(steps, 1):
                    lines.append(f"  {i}. {s}")
                self._append_system("\n".join(lines))
                self._set_spinner(
                    "Writing Python preprocessing script"
                    + (" (may take 1–3 min with local models)…" if "ollama" in
                       (self._conn.label.lower() if self._conn else "") else "…")
                )

        elif node_name == "coder":
            err      = state_delta.get("error")
            deps     = state_delta.get("code_dependencies", [])
            code_len = len(state_delta.get("generated_code", ""))
            retries  = state_delta.get("retries", 0)
            if err:
                self._graph.set_failed("coder")
                self._append_system(f"[{_ts()}] ✗ Coder error: {err}", error=True)
            else:
                self._graph.set_done("coder", retries)
                self._graph.set_active("sandbox_executor")
                msg = f"[{_ts()}] ✓ Script ready ({code_len} chars)"
                if deps:
                    msg += f"  |  pip deps: {', '.join(deps)}"
                self._append_system(msg)
                self._set_spinner(
                    f"Sandbox: installing deps & testing script on sample images…"
                )

        elif node_name in ("sandbox_executor", "local_executor"):
            success = state_delta.get("execution_success")
            stdout  = state_delta.get("execution_stdout", "")
            stderr  = state_delta.get("execution_stderr", "")
            retries = state_delta.get("retries", 0)
            max_ret = self._spin_retries.value()

            if node_name == "sandbox_executor":
                if success:
                    self._graph.set_done("sandbox_executor", retries)
                    self._graph.set_active("local_executor")
                    self._set_spinner(
                        "Sandbox passed — running pipeline on full dataset…"
                    )
                elif retries < max_ret:
                    self._graph.set_retry("sandbox_executor", retries)
                    self._graph.set_active("coder")
                    self._set_spinner(
                        f"Sandbox failed (retry {retries}/{max_ret}) — "
                        "sending error back to coder for self-correction…"
                    )
                else:
                    self._graph.set_failed("sandbox_executor")
            else:
                if success:
                    self._graph.set_done("local_executor")
                    self._graph.set_pipeline_done()
                else:
                    self._graph.set_failed("local_executor")

            icon  = "✓" if success else "✗"
            label = _NODE_LABELS.get(node_name, node_name)
            lines = [f"[{_ts()}] {icon} {label}"]
            if not success and retries:
                lines.append(f"  Retry {retries}/{max_ret}")
            if stdout:
                lines.append(f"  stdout:\n{stdout}")
            if stderr:
                lines.append(f"  stderr:\n{stderr}")
            self._append_system("\n".join(lines), error=(not success))

        elif node_name == "terminal_failure":
            self._graph.set_terminal_failure()
            err = state_delta.get("error", "Max retries exhausted.")
            self._append_system(f"[{_ts()}] ✗ Pipeline aborted: {err}", error=True)

    def _on_pipeline_done(self) -> None:
        self._finalize_stream()
        self._spinner_timer.stop()
        self._resource_timer.stop()
        self._spinner_row.setVisible(False)
        self._pipeline_worker = None
        self._btn_stop.setEnabled(False)
        self._graph.set_pipeline_done()
        self._append_system(f"[{_ts()}] ✓ Pipeline complete.")
        self._validate_inputs()
        self.agent_finished.emit()

    def _on_pipeline_error(self, exc: Exception) -> None:
        self._finalize_stream()
        self._spinner_timer.stop()
        self._resource_timer.stop()
        self._spinner_row.setVisible(False)
        self._pipeline_worker = None
        self._btn_stop.setEnabled(False)
        self._append_system(f"[{_ts()}] ERROR: {exc}", error=True)
        self._validate_inputs()

    # ── Spinner / resource ────────────────────────────────────────────────────

    def _set_spinner(self, text: str) -> None:
        self._spinner_text = text

    def _tick_spinner(self) -> None:
        frame = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._spinner_idx += 1
        text = getattr(self, "_spinner_text", "Running…")
        if self._spinner_lbl:
            self._spinner_lbl.setText(f"{frame}  {text}")

    def _update_resources(self) -> None:
        r = _read_resources()
        parts = []
        if r["cpu"] is not None:
            parts.append(f"CPU {r['cpu']:.0f}%")
        if r["ram_mb"] is not None:
            parts.append(f"RAM {r['ram_mb']}MB")
        if r["gpu_pct"] is not None:
            parts.append(f"GPU {r['gpu_pct']}%  VRAM {r['gpu_mem_mb']}MB")
        self._resource_lbl.setText("  ".join(parts))

    # ── Chat ─────────────────────────────────────────────────────────────────

    def _send_chat(self) -> None:
        text = self._input.text().strip()
        if not text or self._chat_worker is not None or self._conn is None:
            return
        self._input.clear()
        self._append_user(text)
        self._history.append({"role": "user", "content": text})
        self._start_chat_call()

    def _clear_chat(self) -> None:
        self._history.clear()
        self._display.clear()

    def _start_chat_call(self) -> None:
        assert self._conn is not None
        self._btn_send.setEnabled(False)
        self._input.setEnabled(False)
        self._banner.set_status(_DOT_YELLOW, f"Thinking… ({self._conn.label})")

        cfg = self._state.config
        system = _SYSTEM_PROMPT
        if cfg:
            ctx = "\n".join([
                "Current viewer.yaml state:",
                f"  project: {cfg.project.name}",
                f"  axis_order: {cfg.viewer.axis_order}",
                f"  label_layers: {[l.name for l in cfg.label_layers]}",
                f"  active_sample: {self._state.active_sample or 'none'}",
            ])
            system = _SYSTEM_PROMPT + "\n\n" + ctx

        self._chat_worker = _llm_call_worker(self._conn, list(self._history), system)
        self._chat_worker.yielded.connect(self._on_chat_reply)
        self._chat_worker.errored.connect(self._on_chat_error)
        self._chat_worker.finished.connect(self._on_chat_done)
        self._chat_worker.start()

    def _on_chat_reply(self, reply: str) -> None:
        self._history.append({"role": "assistant", "content": reply})
        self._append_assistant(reply)
        patches = extract_patches(reply)
        if patches:
            for patch in patches:
                try:
                    self._state.apply_llm_patch(patch)
                    self._append_system(f"Applied config patch: {list(patch.keys())}")
                    self.config_patch_applied.emit(str(list(patch.keys())))
                except Exception as exc:
                    self._append_system(f"Patch failed: {exc}", error=True)

    def _on_chat_error(self, exc: Exception) -> None:
        self._append_system(f"Chat error: {exc}", error=True)
        if self._conn:
            self._banner.set_status(_DOT_RED, f"Error — {self._conn.label}")
        self._chat_worker = None

    def _on_chat_done(self) -> None:
        if self._conn:
            self._banner.set_status(_DOT_GREEN, f"Connected: {self._conn.label}")
        self._btn_send.setEnabled(True)
        self._input.setEnabled(True)
        self._chat_worker = None

    # ── Display helpers ───────────────────────────────────────────────────────

    def _append_user(self, text: str) -> None:
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self._display.appendHtml(
            f'<b style="color:#ffffff;">You:</b>'
            f'<span style="color:#dddddd;"> {safe}</span>'
            f'<hr style="border:0;border-top:1px solid #2a2a2a;">'
        )
        self._display.moveCursor(QTextCursor.MoveOperation.End)

    def _append_assistant(self, text: str) -> None:
        import re
        html_parts = []
        last = 0
        for m in re.finditer(r"```(?:\w+)?\n?(.*?)```", text, re.DOTALL):
            before = text[last:m.start()].strip()
            if before:
                safe = before.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                safe = safe.replace("\n", "<br>")
                html_parts.append(f'<span style="color:#aaddff;">{safe}</span>')
            code = m.group(1).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_parts.append(
                f'<pre style="background:#1a1a2e;color:#88ddff;'
                f'padding:4px;border-radius:3px;margin:2px 0;'
                f'font-size:10px;">{code}</pre>'
            )
            last = m.end()
        tail = text[last:].strip()
        if tail:
            safe = tail.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            safe = safe.replace("\n", "<br>")
            html_parts.append(f'<span style="color:#aaddff;">{safe}</span>')

        body = "".join(html_parts) or (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "<br>")
        )
        self._display.appendHtml(
            f'<b style="color:#aaddff;">Assistant:</b><br>{body}'
            f'<hr style="border:0;border-top:1px solid #2a2a2a;">'
        )
        self._display.moveCursor(QTextCursor.MoveOperation.End)

    def _append_system(self, text: str, error: bool = False) -> None:
        color = "#ff6666" if error else "#666666"
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        self._display.appendHtml(
            f'<span style="color:{color};font-size:10px;font-family:monospace;">'
            f'{safe}</span>'
        )
        self._display.moveCursor(QTextCursor.MoveOperation.End)

    # ── Prefs ─────────────────────────────────────────────────────────────────

    def _load_prefs(self) -> None:
        prefs = _load_prefs()
        yaml_path = prefs.get("yaml_path", "")
        input_dir = prefs.get("input_dir", "")
        retries   = prefs.get("max_retries", 3)

        if yaml_path and Path(yaml_path).is_file():
            self._set_yaml_path(yaml_path)
        if input_dir and Path(input_dir).is_dir():
            self._set_input_dir(input_dir)
        self._spin_retries.setValue(max(1, min(10, int(retries))))

    def _save_prefs_now(self) -> None:
        _save_prefs({
            "yaml_path":   self._yaml_path,
            "input_dir":   self._input_dir,
            "max_retries": self._spin_retries.value(),
        })
