"""
Enforces deterministic scroll navigation:
  Scroll          → step Z axis
  Shift + Scroll  → step T axis

Works by installing a QObject event filter on the vispy canvas native widget.
The filter intercepts QWheelEvents before they reach the camera, applies the
correct dims step, and accepts (consumes) the event to prevent default zoom.
"""
from __future__ import annotations

from typing import Optional

import napari
from qtpy.QtCore import QEvent, QObject
from qtpy.QtGui import QWheelEvent
from qtpy.QtCore import Qt


class ScrollController(QObject):
    """
    Event filter that remaps mouse wheel events to Z/T dimension steps.

    Install on the canvas native widget via:
        controller = ScrollController(viewer)
        controller.install()
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        # Parent to the viewer Qt window so lifetime is managed
        super().__init__(viewer.window._qt_window)
        self._viewer = viewer
        self._installed = False

    def install(self) -> None:
        """Attach the event filter to the vispy canvas native widget."""
        canvas = self._get_canvas()
        if canvas is not None:
            canvas.installEventFilter(self)
            self._installed = True

    def uninstall(self) -> None:
        canvas = self._get_canvas()
        if canvas is not None:
            canvas.removeEventFilter(self)
            self._installed = False

    def _get_canvas(self):
        try:
            # Use private _qt_viewer to avoid FutureWarning in napari 0.6+
            # (qt_viewer property is deprecated; _qt_viewer is the underlying attr)
            return self._viewer.window._qt_viewer.canvas.native
        except AttributeError:
            return None

    # ------------------------------------------------------------------
    # QObject.eventFilter override
    # ------------------------------------------------------------------

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False  # pass through all non-wheel events

        wheel: QWheelEvent = event
        delta = wheel.angleDelta().y()
        if delta == 0:
            return False

        step = 1 if delta < 0 else -1  # negative delta = scroll down = next slice

        modifiers = wheel.modifiers()
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._step_axis("T", step)
        else:
            self._step_axis("Z", step)

        return True  # consume event — prevents camera zoom

    # ------------------------------------------------------------------
    # Dimension stepping
    # ------------------------------------------------------------------

    def _step_axis(self, axis_char: str, step: int) -> None:
        """
        Step the named axis (Z or T) by ±1 in viewer.dims.
        Finds the correct dims index from the current axis order.
        """
        viewer = self._viewer
        axis_idx = self._find_axis_index(axis_char)
        if axis_idx is None:
            return

        current = list(viewer.dims.current_step)
        ndim = len(current)
        if axis_idx >= ndim:
            return

        new_val = current[axis_idx] + step
        # Clamp to valid range
        range_info = viewer.dims.range[axis_idx]
        lo = int(range_info[0])
        hi = int(range_info[1]) - 1
        new_val = max(lo, min(hi, new_val))
        current[axis_idx] = new_val
        viewer.dims.current_step = tuple(current)

    def _find_axis_index(self, axis_char: str) -> Optional[int]:
        """
        Map an axis character (T, Z, C, Y, X) to a dims index.
        Uses viewer.dims.axis_labels if available, else falls back
        to the project axis_order stored as a viewer attribute.
        """
        viewer = self._viewer

        # napari >= 0.5 exposes axis_labels on dims
        try:
            labels = list(viewer.dims.axis_labels)
            for i, lbl in enumerate(labels):
                if lbl.upper() == axis_char.upper():
                    return i
        except AttributeError:
            pass

        # Fallback: use axis_order stored by our plugin (set during image load)
        axis_order: str = getattr(viewer, "_biovision_axis_order", "TZYX")
        for i, ch in enumerate(axis_order.upper()):
            if ch == axis_char.upper():
                return i

        return None


def install_scroll_controller(viewer: napari.Viewer) -> ScrollController:
    """
    Convenience function: create and install a ScrollController on a viewer.
    Returns the controller so callers can hold a reference (keeps it alive).
    """
    ctrl = ScrollController(viewer)
    ctrl.install()
    return ctrl
