# biovision-napari

A napari plugin for multimodal bioimage segmentation workflows.

## Features

- Load and validate a **project YAML** (`viewer.yaml`)
- **Dataset browser** with sample status tracking (`unlabeled`, `in_progress`, `done`, `reviewed`)
- **Lazy loading** of large 3D/4D+ volumes (dask + zarr)
- Deterministic navigation: **Scroll = Z**, **Shift + Scroll = T**
- Multiple editable **label layers** defined in YAML (cells, nuclei, mitochondria, etc.)
- **Versioned mask saving** (`masks/versions/v0001/`)
- **4 comparison slots** (stacked, synced Z/T + camera) for model output comparison
- **Bookmarking** (sample + Z + T + note)
- **Run Agent** integration with live output
- **LLM chat window** — talk to the assistant to modify your workflow; changes applied automatically

### Supported Image Formats

| Format | Extension | Library |
|--------|-----------|---------|
| Zarr | `.zarr` | zarr + dask |
| TIFF | `.tif`, `.tiff` | tifffile |
| OME-TIFF | `.ome.tif`, `.ome.tiff` | tifffile (OME metadata) |
| N5 | `.n5` | zarr (N5Store) |
| HDF5 | `.h5`, `.hdf5` | h5py + dask |
| Zeiss CZI | `.czi` | aicsimageio |
| Leica LIF | `.lif` | aicsimageio |
| Nikon ND2 | `.nd2` | nd2 |

---

## Installation

```bash
pip install -e .
```

Requires Python 3.11+ and napari 0.5+.

---

## Quickstart

### 1. Create a `viewer.yaml` in your project directory

```yaml
project:
  name: "my_project"
  owner: "James DiMartino"

paths:
  dataset_root: "./data"   # each subdirectory = one sample
  masks: "./data/masks"    # GT masks; versions at ./data/masks/versions/

viewer:
  axis_order: "TZYX"       # T=time, Z=depth, Y, X

label_layers:
  - name: "cells"
    color: "#ff4444"
    classes: ["background", "cell"]
  - name: "nuclei"
    color: "#4444ff"
    classes: ["background", "nucleus"]

llm:
  provider: "anthropic"
  model: "claude-opus-4-6"
  api_key_env: "ANTHROPIC_API_KEY"

agent:
  command: "python run_agent.py"
  working_dir: "."
```

### 2. Organise your data

```
data/
├── sample_001/
│   └── volume.tif
├── sample_002/
│   └── volume.zarr
└── sample_003/
    └── scan.nd2
```

### 3. Launch napari and open the plugin

```bash
napari
```

From the **Plugins** menu, select:
- **BioVision Panel** — main workflow interface
- **Comparison Grid** — 4-slot model comparison
- **LLM Chat** — natural language config assistant

### 4. Load your project

Click **"Open Project (viewer.yaml)"** and select your `viewer.yaml`.

The dataset browser will populate with all samples found in `dataset_root/`.

---

## Mask Versioning

Masks are saved with auto-incrementing version folders:

```
data/masks/versions/
├── v0001/
│   ├── sample_001__cells.tif
│   ├── sample_001__nuclei.tif
│   └── manifest.json
├── v0002/
│   └── ...
```

Each `manifest.json` contains:
```json
{
  "version": "v0001",
  "created_at": "2026-02-28T12:00:00+00:00",
  "sample_id": "sample_001",
  "layers": [
    {"layer": "cells", "file": "sample_001__cells.tif", "sha256": "...", "shape": [50, 512, 512], "dtype": "uint32"}
  ]
}
```

---

## LLM Chat — Applying Config Changes

The LLM assistant can modify `viewer.yaml` by embedding a `yaml-patch` block:

````
I'll add a mitochondria layer for you.

```yaml-patch
label_layers:
  - name: "mitochondria"
    color: "#44ff44"
    classes: ["background", "mitochondrion"]
```
````

The plugin detects this block, applies the patch to `viewer.yaml`, and hot-reloads the UI automatically.

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests cover:
- YAML validation (valid, invalid, roundtrip)
- Version folder increment logic
- Mask save/load roundtrip (values, dtype, multi-version, multi-sample)
