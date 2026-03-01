"""
Per-viewer navigation state: current Z, T, active label layer, active class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NavigationState:
    """Current viewer navigation position."""
    z: int = 0
    t: int = 0
    z_max: int = 0
    t_max: int = 0

    def step_z(self, delta: int) -> None:
        self.z = max(0, min(self.z_max, self.z + delta))

    def step_t(self, delta: int) -> None:
        self.t = max(0, min(self.t_max, self.t + delta))


@dataclass
class LabelState:
    """Which label layer and class is currently active for painting."""
    active_layer_name: Optional[str] = None
    active_class_id: int = 1   # 0 = background

    def set_layer(self, name: str) -> None:
        self.active_layer_name = name

    def set_class(self, class_id: int) -> None:
        self.active_class_id = max(0, class_id)
