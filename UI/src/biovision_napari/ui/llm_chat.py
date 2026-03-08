"""
LLM Chat window.

Supports Anthropic, OpenAI, Groq, and Ollama (local).
Includes a connection setup panel so users can configure a provider directly
in the UI without touching environment variables or config files.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
import urllib.request
from typing import Optional

import yaml
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import (
    QComboBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from napari.qt.threading import thread_worker

from biovision_napari.state.project_state import ProjectState


# ---------------------------------------------------------------------------
# Provider catalogue
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "Groq (free — recommended)": {
        "provider": "openai",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com",
        "needs_key": True,
    },
    "Anthropic": {
        "provider": "anthropic",
        "model": "claude-opus-4-6",
        "base_url": None,
        "key_env": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com",
        "needs_key": True,
    },
    "OpenAI": {
        "provider": "openai",
        "model": "gpt-4o",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        "key_url": "https://platform.openai.com/api-keys",
        "needs_key": True,
    },
    "Ollama (local — no key)": {
        "provider": "openai",
        "model": "llama3.2",
        "base_url": "http://localhost:11434/v1",
        "key_env": None,
        "key_url": "https://ollama.com",
        "needs_key": False,
    },
}


# ---------------------------------------------------------------------------
# Ollama model discovery
# ---------------------------------------------------------------------------

_OLLAMA_BASE = "http://localhost:11434"


def _fetch_ollama_models(base_url: str = _OLLAMA_BASE) -> list[str]:
    """Return model names available in Ollama. Empty list on any failure."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


@thread_worker
def _fetch_ollama_models_worker(base_url: str = _OLLAMA_BASE):
    yield _fetch_ollama_models(base_url)


# ---------------------------------------------------------------------------
# Connection detection
# ---------------------------------------------------------------------------

class _Connection:
    """Resolved connection parameters ready to use."""
    def __init__(self, provider, model, api_key, base_url, label):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.label = label   # human-readable string for the status bar


def _detect_connection(cfg=None) -> Optional[_Connection]:
    """
    Return a live _Connection or None.
    Priority: configured provider → Groq env → Anthropic env → Ollama local.
    """
    # 1. Try the project-configured provider
    if cfg is not None:
        key = os.environ.get(cfg.llm.api_key_env, "")
        if key:
            return _Connection(
                provider=cfg.llm.provider,
                model=cfg.llm.model,
                api_key=key,
                base_url=cfg.llm.base_url,
                label=f"{cfg.llm.provider} / {cfg.llm.model}",
            )

    # 2. Groq free tier
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if groq_key:
        p = _PROVIDERS["Groq (free — recommended)"]
        return _Connection("openai", p["model"], groq_key,
                           p["base_url"], "Groq / llama-3.3-70b (guest)")

    # 3. Anthropic
    ant_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if ant_key:
        p = _PROVIDERS["Anthropic"]
        return _Connection("anthropic", p["model"], ant_key,
                           None, "Anthropic / claude-opus-4-6 (guest)")

    # 4. OpenAI
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if oai_key:
        p = _PROVIDERS["OpenAI"]
        return _Connection("openai", p["model"], oai_key,
                           None, "OpenAI / gpt-4o (guest)")

    # 5. Ollama running locally
    try:
        p = _PROVIDERS["Ollama (local — no key)"]
        models = _fetch_ollama_models()
        model = models[0] if models else p["model"]
        return _Connection("openai", model, "ollama",
                           p["base_url"], f"Ollama / {model} (local)")
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# LLM backend calls
# ---------------------------------------------------------------------------

def _call_anthropic(model: str, api_key: str, messages: list[dict], system: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model, max_tokens=4096, system=system, messages=messages,
    )
    return response.content[0].text


def _call_openai_compat(
    model: str, api_key: str, base_url: Optional[str],
    messages: list[dict], system: str,
) -> str:
    import openai
    kwargs: dict = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = openai.OpenAI(**kwargs)
    full = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(model=model, messages=full, max_tokens=4096)
    return resp.choices[0].message.content


@thread_worker
def _llm_call_worker(conn: _Connection, messages: list[dict], system_prompt: str):
    """Background thread: call the LLM and yield the reply string."""
    if conn.provider == "anthropic":
        reply = _call_anthropic(conn.model, conn.api_key, messages, system_prompt)
    else:
        reply = _call_openai_compat(
            conn.model, conn.api_key, conn.base_url, messages, system_prompt
        )
    yield reply


@thread_worker
def _test_connection_worker(conn: _Connection):
    """Send a minimal ping to verify the connection is live."""
    ping = [{"role": "user", "content": "Reply with just: ok"}]
    if conn.provider == "anthropic":
        _call_anthropic(conn.model, conn.api_key, ping, "")
    else:
        _call_openai_compat(conn.model, conn.api_key, conn.base_url, ping, "")
    yield True


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

_PATCH_RE = re.compile(r"```yaml-patch\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_patches(text: str) -> list[dict]:
    patches = []
    for match in _PATCH_RE.finditer(text):
        try:
            patch = yaml.safe_load(match.group(1))
            if isinstance(patch, dict):
                patches.append(patch)
        except yaml.YAMLError:
            pass
    return patches


# ---------------------------------------------------------------------------
# Setup panel (collapsible)
# ---------------------------------------------------------------------------

class _SetupPanel(QGroupBox):
    """
    Collapsible widget shown when no LLM is configured.
    Lets the user pick a provider, paste an API key, and connect in one click.
    """

    connected = Signal(_Connection)  # emitted when test passes

    def __init__(self, parent=None):
        super().__init__("Connect to an LLM", parent)
        self._test_worker = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        self._fetch_worker = None  # background Ollama model fetch

        # Provider picker
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Provider:"))
        self._combo = QComboBox()
        for name in _PROVIDERS:
            self._combo.addItem(name)
        self._combo.currentTextChanged.connect(self._on_provider_changed)
        row1.addWidget(self._combo, stretch=1)
        layout.addLayout(row1)

        # Model row — combobox for Ollama, line-edit for cloud providers
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        row_model.addWidget(self._model_combo, stretch=1)
        self._btn_refresh = QPushButton("↻")
        self._btn_refresh.setFixedWidth(28)
        self._btn_refresh.setToolTip("Refresh model list from Ollama")
        self._btn_refresh.clicked.connect(self._refresh_ollama_models)
        row_model.addWidget(self._btn_refresh)
        self._model_edit = QLineEdit()
        row_model.addWidget(self._model_edit, stretch=1)
        layout.addLayout(row_model)

        # Key entry
        row2 = QHBoxLayout()
        self._lbl_key = QLabel("API Key:")
        row2.addWidget(self._lbl_key)
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("Paste your API key here…")
        row2.addWidget(self._key_input, stretch=1)
        self._btn_show = QPushButton("Show")
        self._btn_show.setFixedWidth(48)
        self._btn_show.setCheckable(True)
        self._btn_show.toggled.connect(self._toggle_key_visibility)
        row2.addWidget(self._btn_show)
        layout.addLayout(row2)

        # Action buttons
        row3 = QHBoxLayout()
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        row3.addWidget(self._btn_connect)

        self._lbl_hint = QLabel()
        self._lbl_hint.setOpenExternalLinks(True)
        self._lbl_hint.setStyleSheet("color: #5599ff; font-size: 10px;")
        row3.addWidget(self._lbl_hint, stretch=1)
        layout.addLayout(row3)

        self._lbl_status = QLabel("")
        self._lbl_status.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._lbl_status)

        self._on_provider_changed(self._combo.currentText())

    # ------------------------------------------------------------------

    def _on_provider_changed(self, name: str) -> None:
        p = _PROVIDERS[name]
        needs_key = p["needs_key"]
        is_ollama = not needs_key  # currently only Ollama is keyless

        # Show the right model widget
        self._model_combo.setVisible(is_ollama)
        self._btn_refresh.setVisible(is_ollama)
        self._model_edit.setVisible(not is_ollama)
        if not is_ollama:
            self._model_edit.setText(p["model"])

        self._lbl_key.setVisible(needs_key)
        self._key_input.setVisible(needs_key)
        self._btn_show.setVisible(needs_key)
        self._key_input.clear()
        url = p["key_url"]
        if needs_key:
            self._lbl_hint.setText(f'<a href="{url}">Get a free key →</a>')
        else:
            self._lbl_hint.setText(
                f'Make sure Ollama is running. '
                f'<a href="{url}">Install Ollama →</a>'
            )
            self._refresh_ollama_models()
        self._lbl_status.setText("")

    def _refresh_ollama_models(self) -> None:
        """Fetch available models from Ollama in a background thread."""
        if self._fetch_worker is not None:
            return
        self._model_combo.clear()
        self._model_combo.addItem("Fetching models…")
        self._model_combo.setEnabled(False)
        self._btn_refresh.setEnabled(False)
        self._fetch_worker = _fetch_ollama_models_worker()
        self._fetch_worker.yielded.connect(self._on_models_fetched)
        self._fetch_worker.finished.connect(self._on_fetch_done)
        self._fetch_worker.start()

    def _on_models_fetched(self, models: list) -> None:
        self._model_combo.clear()
        if models:
            for m in models:
                self._model_combo.addItem(m)
            self._lbl_status.setText(f"{len(models)} model(s) found")
            self._lbl_status.setStyleSheet("color: #44ff88; font-size: 10px;")
        else:
            self._model_combo.addItem("llama3.2")  # sensible fallback
            self._lbl_status.setText("Ollama not reachable — showing default model")
            self._lbl_status.setStyleSheet("color: #ffaa44; font-size: 10px;")

    def _on_fetch_done(self) -> None:
        self._model_combo.setEnabled(True)
        self._btn_refresh.setEnabled(True)
        self._fetch_worker = None

    def _toggle_key_visibility(self, visible: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        self._key_input.setEchoMode(mode)
        self._btn_show.setText("Hide" if visible else "Show")

    def _on_connect(self) -> None:
        name = self._combo.currentText()
        p = _PROVIDERS[name]

        if p["needs_key"]:
            key = self._key_input.text().strip()
            if not key:
                self._lbl_status.setText("⚠ Paste your API key first.")
                self._lbl_status.setStyleSheet("color: #ffaa44; font-size: 10px;")
                return
            model = self._model_edit.text().strip() or p["model"]
        else:
            key = "ollama"
            model = self._model_combo.currentText().strip() or p["model"]

        conn = _Connection(
            provider=p["provider"],
            model=model,
            api_key=key,
            base_url=p["base_url"],
            label=f"{name} / {model}",
        )

        self._btn_connect.setEnabled(False)
        self._lbl_status.setText("Testing connection…")
        self._lbl_status.setStyleSheet("color: #888; font-size: 10px;")

        self._test_worker = _test_connection_worker(conn)
        self._test_worker.yielded.connect(lambda _: self._on_test_ok(conn, p))
        self._test_worker.errored.connect(self._on_test_fail)
        self._test_worker.start()

    def _on_test_ok(self, conn: _Connection, p: dict) -> None:
        # Persist key in environment for this session
        if p["needs_key"] and p["key_env"]:
            os.environ[p["key_env"]] = conn.api_key
        self._lbl_status.setText("✓ Connected!")
        self._lbl_status.setStyleSheet("color: #44ff88; font-size: 10px;")
        self._btn_connect.setEnabled(True)
        self._test_worker = None
        self.connected.emit(conn)

    def _on_test_fail(self, exc: Exception) -> None:
        self._lbl_status.setText(f"✗ {exc}")
        self._lbl_status.setStyleSheet("color: #ff6666; font-size: 10px;")
        self._btn_connect.setEnabled(True)
        self._test_worker = None


# ---------------------------------------------------------------------------
# Status banner
# ---------------------------------------------------------------------------

_DOT_GREEN  = '<span style="color:#44ff88; font-size:14px;">●</span>'
_DOT_YELLOW = '<span style="color:#ffcc44; font-size:14px;">●</span>'
_DOT_RED    = '<span style="color:#ff4444; font-size:14px;">●</span>'


class _StatusBanner(QWidget):
    setup_toggled = Signal(bool)   # True = show setup panel

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        self._dot = QLabel(_DOT_RED)
        self._dot.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._dot)

        self._lbl = QLabel("No LLM connected")
        self._lbl.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._lbl, stretch=1)

        self._btn_setup = QPushButton("Set up ▾")
        self._btn_setup.setCheckable(True)
        self._btn_setup.setFixedWidth(72)
        self._btn_setup.toggled.connect(self.setup_toggled)
        layout.addWidget(self._btn_setup)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444;")

    def set_status(self, dot: str, text: str) -> None:
        self._dot.setText(dot)
        self._lbl.setText(text)

    def hide_setup_button(self) -> None:
        self._btn_setup.setChecked(False)
        self._btn_setup.setVisible(False)


# ---------------------------------------------------------------------------
# Main chat widget
# ---------------------------------------------------------------------------

class LLMChatWidget(QWidget):
    """
    Interactive LLM chat dock widget.

    On first open, detects available providers and shows a setup panel if none
    are configured. Once connected, the panel collapses and the chat activates.
    """

    config_patch_applied = Signal(str)

    def __init__(self, state: ProjectState, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._state = state
        self._conn: Optional[_Connection] = None
        self._history: list[dict] = []
        self._worker = None
        self._build_ui()
        self._state.config_changed.connect(self._on_config_changed)
        # Detect any already-available connection on startup
        self._refresh_connection()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Status banner
        self._banner = _StatusBanner()
        self._banner.setup_toggled.connect(self._on_setup_toggled)
        layout.addWidget(self._banner)

        # Setup panel (hidden when connected)
        self._setup = _SetupPanel()
        self._setup.connected.connect(self._on_connected)
        self._setup.setVisible(False)
        layout.addWidget(self._setup)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #333;")
        layout.addWidget(line)

        # Chat display
        self._display = QPlainTextEdit()
        self._display.setReadOnly(True)
        self._display.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._display)

        # Input row
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Ask BioVision assistant…")
        self._input.returnPressed.connect(self._send)
        self._input.setEnabled(False)
        input_row.addWidget(self._input)

        self._btn_send = QPushButton("Send")
        self._btn_send.clicked.connect(self._send)
        self._btn_send.setEnabled(False)
        input_row.addWidget(self._btn_send)

        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear)
        input_row.addWidget(btn_clear)

        layout.addLayout(input_row)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _refresh_connection(self) -> None:
        """Check for an available connection and update the UI accordingly."""
        cfg = self._state.config
        conn = _detect_connection(cfg)
        if conn:
            self._on_connected(conn)
        else:
            self._banner.set_status(_DOT_RED, "No LLM connected — click Set up to configure")
            self._setup.setVisible(True)
            self._banner._btn_setup.setChecked(True)
            self._input.setEnabled(False)
            self._btn_send.setEnabled(False)

    def _on_connected(self, conn: _Connection) -> None:
        self._conn = conn
        self._banner.set_status(_DOT_GREEN, f"Connected: {conn.label}")
        self._setup.setVisible(False)
        self._banner._btn_setup.setChecked(False)
        self._input.setEnabled(True)
        self._btn_send.setEnabled(True)
        if not self._history:
            self._append_message(
                "Assistant",
                f"Hi! I'm the BioVision assistant running on {conn.label}.\n"
                "I can help you configure your segmentation workflow. "
                "Ask me anything, or describe a change you'd like to make.",
            )

    def _on_setup_toggled(self, checked: bool) -> None:
        self._setup.setVisible(checked)

    # ------------------------------------------------------------------
    # Config / state changes
    # ------------------------------------------------------------------

    def _on_config_changed(self, config) -> None:
        if config and self._conn is None:
            self._refresh_connection()

    def accept_agent_config(
        self, provider: str, model: str, api_key: str, base_url: str
    ) -> None:
        """
        Called by AgentPanel (via llm_config_ready signal) when the pipeline
        starts.  Synchronises the chat widget to the same LLM connection so the
        user only needs to configure credentials once.
        """
        if provider == "anthropic" and api_key:
            conn = _Connection(
                provider="anthropic",
                model=model or "claude-opus-4-6",
                api_key=api_key,
                base_url=None,
                label=f"Anthropic / {model or 'claude-opus-4-6'} (from Agent tab)",
            )
            self._on_connected(conn)
        elif provider == "ollama":
            conn = _Connection(
                provider="openai",
                model=model or "llama3.2",
                api_key="ollama",
                base_url=(base_url or "http://localhost:11434") + "/v1",
                label=f"Ollama / {model or 'llama3.2'} (from Agent tab)",
            )
            self._on_connected(conn)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def _send(self) -> None:
        text = self._input.text().strip()
        if not text or self._worker is not None or self._conn is None:
            return
        self._input.clear()
        self._append_message("You", text)
        self._history.append({"role": "user", "content": text})
        self._start_llm_call()

    def _clear(self) -> None:
        self._history.clear()
        self._display.clear()

    def _start_llm_call(self) -> None:
        assert self._conn is not None
        self._btn_send.setEnabled(False)
        self._input.setEnabled(False)
        self._banner.set_status(_DOT_YELLOW, f"Thinking… ({self._conn.label})")

        cfg = self._state.config
        system = ""
        if cfg:
            system = cfg.llm.system_prompt + "\n\n" + self._build_context_block()

        self._worker = _llm_call_worker(self._conn, list(self._history), system)
        self._worker.yielded.connect(self._on_llm_reply)
        self._worker.errored.connect(self._on_llm_error)
        self._worker.finished.connect(self._on_llm_done)
        self._worker.start()

    def _build_context_block(self) -> str:
        cfg = self._state.config
        if cfg is None:
            return ""
        return "\n".join([
            "Current viewer.yaml state:",
            f"  project: {cfg.project.name}",
            f"  axis_order: {cfg.viewer.axis_order}",
            f"  label_layers: {[l.name for l in cfg.label_layers]}",
            f"  active_sample: {self._state.active_sample or 'none'}",
        ])

    def _on_llm_reply(self, reply: str) -> None:
        self._history.append({"role": "assistant", "content": reply})
        self._append_message("Assistant", reply)
        patches = extract_patches(reply)
        if patches:
            applied = []
            for patch in patches:
                try:
                    self._state.apply_llm_patch(patch)
                    applied.append(str(list(patch.keys())))
                except Exception as exc:
                    self._append_message("System", f"Patch failed: {exc}", error=True)
            if applied:
                summary = f"Applied config changes: {', '.join(applied)}"
                self._append_message("System", summary)
                self.config_patch_applied.emit(summary)

    def _on_llm_error(self, exc: Exception) -> None:
        self._append_message("System", f"Error: {exc}", error=True)
        self._banner.set_status(_DOT_RED, f"Error — {self._conn.label if self._conn else ''}")
        self._worker = None

    def _on_llm_done(self) -> None:
        if self._conn:
            self._banner.set_status(_DOT_GREEN, f"Connected: {self._conn.label}")
        self._btn_send.setEnabled(True)
        self._input.setEnabled(True)
        self._worker = None

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _append_message(self, sender: str, text: str, error: bool = False) -> None:
        color = "#ff6666" if error else ("#aaddff" if sender == "Assistant" else "#ffffff")
        wrapped = textwrap.fill(text, width=80, subsequent_indent="  ")
        html = (
            f'<b style="color:{color};">{sender}:</b>'
            f'<pre style="margin:0; white-space:pre-wrap; color:{color};">{wrapped}</pre>'
            f'<hr style="border:0; border-top:1px solid #333;">'
        )
        self._display.appendHtml(html)
        self._display.moveCursor(QTextCursor.MoveOperation.End)
