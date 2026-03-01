"""Pydantic schema for viewer.yaml validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class ProjectMeta(BaseModel):
    name: str = "untitled"
    description: str = ""
    owner: str = ""
    created: str = ""
    version: str = "0.1"


class Paths(BaseModel):
    dataset_root: str = "./data"
    masks: str = "./data/masks"
    output_root: str = "./runs"
    cache_root: str = "./.cache"


class ViewerConfig(BaseModel):
    axis_order: str = "TZYX"
    default_colormap: str = "gray"

    @field_validator("axis_order")
    @classmethod
    def validate_axis_order(cls, v: str) -> str:
        allowed = set("TCZYX")
        if not set(v.upper()).issubset(allowed):
            raise ValueError(f"axis_order contains unknown axes: {v!r}. Allowed: T C Z Y X")
        if len(v) != len(set(v)):
            raise ValueError(f"axis_order has duplicate axes: {v!r}")
        return v.upper()


class SampleConfig(BaseModel):
    discovery: Literal["subdirectory"] = "subdirectory"
    status_file: str = "status.json"


class LabelLayer(BaseModel):
    name: str
    color: str = "red"
    opacity: float = 0.5
    classes: list[str] = Field(default_factory=lambda: ["background"])

    @field_validator("classes")
    @classmethod
    def background_first(cls, v: list[str]) -> list[str]:
        if not v:
            return ["background"]
        return v


class IOConfig(BaseModel):
    supported_formats: list[str] = Field(
        default_factory=lambda: [
            "zarr", "tif", "tiff", "n5", "h5", "hdf5",
            "czi", "lif", "nd2", "ome.tiff",
        ]
    )
    image_glob: str = "**/*.tif"
    axes_override: Optional[str] = None


class AgentConfig(BaseModel):
    command: str = "python run_agent.py"
    working_dir: str = "."


class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai", "ollama"] = "anthropic"
    model: str = "claude-opus-4-6"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: Optional[str] = None
    system_prompt: str = (
        "You are a BioVision assistant. Help the user configure their segmentation "
        "workflow. You can suggest and apply changes to viewer.yaml. When you want to "
        "apply a config change, output a JSON block tagged ```yaml-patch with the "
        "nested keys to update."
    )


class Bookmark(BaseModel):
    sample_id: str
    z: int = 0
    t: int = 0
    note: str = ""


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class ViewerConfig_Root(BaseModel):
    """Root model for viewer.yaml."""

    project: ProjectMeta = Field(default_factory=ProjectMeta)
    paths: Paths = Field(default_factory=Paths)
    viewer: ViewerConfig = Field(default_factory=ViewerConfig)
    samples: SampleConfig = Field(default_factory=SampleConfig)
    label_layers: list[LabelLayer] = Field(
        default_factory=lambda: [
            LabelLayer(name="cells", color="#ff0000", classes=["background", "cell"]),
            LabelLayer(name="nuclei", color="#0000ff", classes=["background", "nucleus"]),
        ]
    )
    io: IOConfig = Field(default_factory=IOConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    bookmarks: list[Bookmark] = Field(default_factory=list)

    @model_validator(mode="after")
    def label_layers_not_empty(self) -> "ViewerConfig_Root":
        if not self.label_layers:
            raise ValueError("viewer.yaml must define at least one label_layer")
        return self


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def load_viewer_yaml(path: str | Path) -> ViewerConfig_Root:
    """Parse and validate viewer.yaml, returning a ViewerConfig_Root."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"viewer.yaml not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ViewerConfig_Root.model_validate(raw or {})


def save_viewer_yaml(config: ViewerConfig_Root, path: str | Path) -> None:
    """Serialize config back to viewer.yaml (round-trip safe)."""
    path = Path(path)
    data = config.model_dump(mode="json", exclude_none=False)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, allow_unicode=True, sort_keys=False, default_flow_style=False)


def apply_patch(config: ViewerConfig_Root, patch: dict[str, Any]) -> ViewerConfig_Root:
    """
    Deep-merge a patch dict into config and return a new validated instance.
    Used by the LLM chat to apply suggested changes.
    """
    current = config.model_dump(mode="json")
    _deep_merge(current, patch)
    return ViewerConfig_Root.model_validate(current)


def _deep_merge(base: dict, patch: dict) -> None:
    for key, value in patch.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
