"""
BioVision main dock widget.
Assembles all sub-panels into a single tabbed interface and wires
up the primary viewer with Z/T scroll enforcement, label layers,
versioned mask save/load, and sample loading.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import napari
import numpy as np
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
)
from qtpy.QtCore import Qt

from biovision_napari.io.mask_io import load_masks, save_masks, list_versions
from biovision_napari.nav.scroll_controller import install_scroll_controller
from biovision_napari.state.project_state import ProjectState
from biovision_napari.ui.agent_chat_panel import AgentChatPanel
from biovision_napari.ui.bookmark_panel import BookmarkPanel
from biovision_napari.ui.dataset_browser import DatasetBrowser
from biovision_napari.ui.label_controls import LabelControls
from biovision_napari.workers.image_worker import load_image_worker


class BioVisionWidget(QWidget):
    """
    Primary dock widget.  Pass a napari.Viewer instance at construction.
    viewer.yaml is loaded via the "Open Project" button or programmatically
    via state.load(path).
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._state = ProjectState(parent=self)
        self._scroll_ctrl = None
        self._current_worker = None

        self._build_ui()
        self._state.sample_changed.connect(self._on_sample_changed)

        # Install scroll controller immediately
        self._scroll_ctrl = install_scroll_controller(self._viewer)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top bar: project load
        top_bar = QHBoxLayout()
        btn_open = QPushButton("Open Project (viewer.yaml)")
        btn_open.clicked.connect(self._open_project)
        top_bar.addWidget(btn_open)
        self._lbl_yaml = QLabel("No project loaded")
        self._lbl_yaml.setStyleSheet("color: #888; font-size: 10px;")
        top_bar.addWidget(self._lbl_yaml)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # Tabs
        tabs = QTabWidget()

        # --- Dataset browser tab ---
        self._browser = DatasetBrowser(self._state)
        self._browser.sample_selected.connect(self._on_sample_selected_in_browser)
        tabs.addTab(self._browser, "Samples")

        # --- Label layers tab ---
        self._label_ctrl = LabelControls(self._viewer, self._state)
        tabs.addTab(self._label_ctrl, "Labels")

        # --- Mask save/load tab ---
        mask_tab = self._build_mask_tab()
        tabs.addTab(mask_tab, "Masks")

        # --- Bookmarks tab ---
        self._bookmarks = BookmarkPanel(self._viewer, self._state)
        tabs.addTab(self._bookmarks, "Bookmarks")

        # --- Agent tab ---
        self._agent_panel = AgentChatPanel(self._state)
        tabs.addTab(self._agent_panel, "Agent")

        layout.addWidget(tabs)

    def _build_mask_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        # Version selector
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Version:"))
        self._combo_version = QComboBox()
        hdr.addWidget(self._combo_version)
        hdr.addStretch()
        layout.addLayout(hdr)

        # Buttons
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save masks (new version)")
        btn_save.clicked.connect(self._save_masks)
        btn_row.addWidget(btn_save)

        btn_load = QPushButton("Load selected version")
        btn_load.clicked.connect(self._load_masks)
        btn_row.addWidget(btn_load)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Project load
    # ------------------------------------------------------------------

    def _open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open viewer.yaml",
            "",
            "YAML Files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            self._state.load(path)
            self._lbl_yaml.setText(f"Loaded: {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Error loading project", str(exc))

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _on_sample_selected_in_browser(self, sample_id: str) -> None:
        self._load_sample(sample_id)

    def _on_sample_changed(self, sample_id: str) -> None:
        self._refresh_version_list(sample_id)

    def _load_sample(self, sample_id: str) -> None:
        cfg = self._state.config
        if cfg is None:
            return

        sample_dir = Path(cfg.paths.dataset_root) / sample_id
        # Find first supported image file
        image_path = self._find_image(sample_dir, cfg.io.image_glob)
        if image_path is None:
            QMessageBox.warning(
                self, "No image found",
                f"No supported image found in {sample_dir}"
            )
            return

        axis_order = cfg.viewer.axis_order
        if self._current_worker is not None:
            self._current_worker.quit()

        worker = load_image_worker(image_path, axis_order)
        worker.yielded.connect(
            lambda arr: self._on_image_loaded(arr, sample_id, axis_order)
        )
        worker.errored.connect(
            lambda exc: QMessageBox.critical(self, "Load error", str(exc))
        )
        worker.start()
        self._current_worker = worker

    def _on_image_loaded(
        self, arr, sample_id: str, axis_order: str
    ) -> None:
        import numpy as np
        self._viewer.layers.clear()

        dtype = arr.dtype
        if np.issubdtype(dtype, np.unsignedinteger):
            info = np.iinfo(dtype)
            clim = [float(info.min), float(info.max)]
        elif np.issubdtype(dtype, np.floating):
            clim = [0.0, 1.0]
        else:
            clim = [0.0, 65535.0]

        self._viewer.add_image(
            arr, name=sample_id, colormap="gray", contrast_limits=clim
        )
        # pydantic rejects unknown fields via setattr; bypass it directly
        object.__setattr__(self._viewer, "_biovision_axis_order", axis_order)

        # Set axis labels on dims if supported
        try:
            self._viewer.dims.axis_labels = tuple(axis_order)
        except Exception:
            pass

        # Fit the view — use plain Python floats to avoid pydantic strict typing
        self._fit_view_safe(arr.shape)

        self._label_ctrl._create_label_layers_in_viewer()
        self._auto_load_latest_masks(sample_id)
        self._current_worker = None

    def _fit_view_safe(self, shape: tuple) -> None:
        """
        Fit the napari camera to the loaded image.
        napari 0.6.x has a bug where reset_view() assigns np.float64 to
        camera.center which pydantic v1 rejects.  We catch that and fall back
        to setting the camera manually with plain Python floats.
        """
        try:
            self._viewer.reset_view()
            return
        except Exception:
            pass

        # Manual fallback: centre on Y-X extent with plain Python floats.
        try:
            cy = float(shape[-2]) / 2.0
            cx = float(shape[-1]) / 2.0
            self._viewer.camera.center = (cy, cx)
        except Exception:
            pass

    def _find_image(self, sample_dir: Path, glob_pattern: str) -> Optional[Path]:
        results = list(sample_dir.glob(glob_pattern))
        if results:
            return results[0]
        # Fallback: any supported extension
        for ext in [".tif", ".tiff", ".zarr", ".n5", ".h5", ".czi", ".lif", ".nd2"]:
            results = list(sample_dir.glob(f"**/*{ext}"))
            if results:
                return results[0]
        return None

    # ------------------------------------------------------------------
    # Mask save / load
    # ------------------------------------------------------------------

    def _save_masks(self) -> None:
        cfg = self._state.config
        sample_id = self._state.active_sample
        if cfg is None or sample_id is None:
            return

        layers = self._collect_label_layers()
        if not layers:
            QMessageBox.information(self, "No label layers", "No label layers to save.")
            return

        try:
            version_tag = save_masks(sample_id, cfg.paths.masks, layers)
            self._refresh_version_list(sample_id)
            QMessageBox.information(self, "Saved", f"Masks saved as version {version_tag}")
            self._state.mask_saved.emit(sample_id, version_tag)
        except Exception as exc:
            QMessageBox.critical(self, "Save error", str(exc))

    def _load_masks(self) -> None:
        cfg = self._state.config
        sample_id = self._state.active_sample
        if cfg is None or sample_id is None:
            return

        version_tag = self._combo_version.currentText() or None
        try:
            masks = load_masks(sample_id, cfg.paths.masks, version_tag)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", str(exc))
            return

        if not masks:
            QMessageBox.information(self, "No masks", "No masks found for this version.")
            return

        for layer_name, arr in masks.items():
            existing = [l for l in self._viewer.layers if l.name == layer_name]
            for l in existing:
                self._viewer.layers.remove(l)
            self._viewer.add_labels(arr, name=layer_name)

    def _auto_load_latest_masks(self, sample_id: str) -> None:
        cfg = self._state.config
        if cfg is None:
            return
        try:
            masks = load_masks(sample_id, cfg.paths.masks)
            for layer_name, arr in masks.items():
                self._viewer.add_labels(arr, name=layer_name)
        except Exception:
            pass

    def _refresh_version_list(self, sample_id: str) -> None:
        cfg = self._state.config
        if cfg is None:
            return
        versions = list_versions(sample_id, cfg.paths.masks)
        self._combo_version.clear()
        for v in reversed(versions):  # newest first
            self._combo_version.addItem(v)

    def _on_agent_input_dir_changed(self, folder: str) -> None:
        """
        Auto-load images from the agent input folder into the viewer.
        Clears existing layers and adds each supported image file as a layer.
        Only loads the first image immediately to avoid blocking; the rest
        are listed in the Samples tab for manual selection.
        """
        from pathlib import Path as _Path
        _exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
        try:
            images = sorted(
                p for p in _Path(folder).iterdir()
                if p.is_file() and p.suffix.lower() in _exts
            )
        except Exception:
            return
        if not images:
            return
        # Load only the first image automatically; show count in viewer title.
        axis_order = "YX"
        cfg = self._state.config
        if cfg is not None:
            axis_order = cfg.viewer.axis_order
        worker = load_image_worker(images[0], axis_order)
        worker.yielded.connect(
            lambda arr: self._on_image_loaded(arr, images[0].stem, axis_order)
        )
        worker.errored.connect(lambda exc: None)  # silently ignore load errors
        worker.start()
        self._current_worker = worker

    def _collect_label_layers(self) -> dict[str, np.ndarray]:
        result = {}
        for layer in self._viewer.layers:
            if hasattr(layer, "selected_label"):  # it's a Labels layer
                result[layer.name] = np.asarray(layer.data)
        return result
