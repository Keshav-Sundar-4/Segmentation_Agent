"""
Central application state. Holds the parsed viewer.yaml config and the
list of discovered samples. Emits Qt signals when state changes so that
all widgets can react without direct coupling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy.QtCore import QObject, Signal

from biovision_napari.io.yaml_schema import (
    ViewerConfig_Root,
    load_viewer_yaml,
    save_viewer_yaml,
    apply_patch,
)


class ProjectState(QObject):
    """
    Singleton-like state object passed to all widgets.
    Signals fire whenever state changes so widgets can update themselves.
    """

    # Fired after viewer.yaml is loaded or reloaded
    config_changed = Signal(object)  # payload: ViewerConfig_Root

    # Fired after the active sample changes
    sample_changed = Signal(str)  # payload: sample_id

    # Fired after a mask version is saved
    mask_saved = Signal(str, str)  # sample_id, version_tag

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._config: Optional[ViewerConfig_Root] = None
        self._yaml_path: Optional[Path] = None
        self._active_sample: Optional[str] = None

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @property
    def config(self) -> Optional[ViewerConfig_Root]:
        return self._config

    @property
    def yaml_path(self) -> Optional[Path]:
        return self._yaml_path

    def load(self, yaml_path: str | Path) -> None:
        """Load and validate viewer.yaml from disk."""
        self._yaml_path = Path(yaml_path)
        self._config = load_viewer_yaml(self._yaml_path)
        self.config_changed.emit(self._config)

    def reload(self) -> None:
        """Re-read viewer.yaml from disk (hot-reload after LLM patch)."""
        if self._yaml_path is None:
            return
        self._config = load_viewer_yaml(self._yaml_path)
        self.config_changed.emit(self._config)

    def apply_llm_patch(self, patch: dict) -> None:
        """
        Apply a partial YAML patch (from LLM chat), persist to disk, and
        emit config_changed so all widgets hot-reload.
        """
        if self._config is None or self._yaml_path is None:
            raise RuntimeError("No viewer.yaml loaded")
        self._config = apply_patch(self._config, patch)
        save_viewer_yaml(self._config, self._yaml_path)
        self.config_changed.emit(self._config)

    def save_config(self) -> None:
        """Persist current in-memory config back to viewer.yaml."""
        if self._config is None or self._yaml_path is None:
            return
        save_viewer_yaml(self._config, self._yaml_path)

    # ------------------------------------------------------------------
    # Active sample
    # ------------------------------------------------------------------

    @property
    def active_sample(self) -> Optional[str]:
        return self._active_sample

    def set_active_sample(self, sample_id: str) -> None:
        if sample_id != self._active_sample:
            self._active_sample = sample_id
            self.sample_changed.emit(sample_id)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def dataset_root(self) -> Optional[Path]:
        if self._config is None:
            return None
        return Path(self._config.paths.dataset_root)

    @property
    def masks_root(self) -> Optional[Path]:
        if self._config is None:
            return None
        return Path(self._config.paths.masks)

    @property
    def versions_root(self) -> Optional[Path]:
        if self.masks_root is None:
            return None
        return self.masks_root / "versions"

    @property
    def axis_order(self) -> str:
        if self._config is None:
            return "TZYX"
        return self._config.viewer.axis_order
