# AGENT_INSTRUCTIONS.md

## napari Multimodal Segmentation GUI (Frontend-Oriented)

------------------------------------------------------------------------

# Mission

Build a **napari plugin** that provides a **general-purpose GUI for
multimodal segmentation workflows**.

This GUI is the **primary human interface** to the broader agentic
segmentation system.

The frontend must:

-   Load and validate a **project YAML**
-   Browse samples with statuses
-   Display large 3D/4D+ multimodal volumes using **lazy loading
    (dask/zarr)**
-   Enforce deterministic navigation:
    -   **Scroll = Z**
    -   **Shift + Scroll = T**
-   Support multiple editable **label layers** defined in YAML
-   Save masks using a **version folder scheme**
-   Display up to **4 user-selectable model comparison slots**
-   Provide bookmarking
-   Provide "Run Agent" integration (stub execution acceptable in v0)

This is **GUI-only scope**. The segmentation models and backend agent
logic are out of scope beyond command execution and output refresh.

------------------------------------------------------------------------

# Non-Negotiable UX Constraints

1.  Scroll wheel always controls **Z**
2.  Shift + scroll always controls **T**
3.  Label layers are defined by YAML (e.g. `cells`, `nuclei`,
    `mitochondria`, `organelles`)
4.  Classes are shared across layers
5.  v0 interpolation is **Z-only**
6.  Masks saved as **one file per label layer**
7.  Masks saved inside **version folders**
8.  4 comparison slots are user-selectable and may be hidden

------------------------------------------------------------------------

# Acceptance Tests (Definition of Done)

The plugin is complete when the following work:

## A. Project Load + Dataset Browser

-   YAML loads and validates
-   Dataset browser lists:
    -   sample_id
    -   status
    -   modality
    -   dims
    -   has_gt_mask
    -   models_available
-   Clicking a sample loads it into the viewer

## B. Viewing + Navigation

-   Scroll changes Z
-   Shift+Scroll changes T
-   Z/T positions are synchronized across visible comparison views
-   Pan/zoom is linked across views

## C. Mask Editing + Saving

-   Editable GT view shows label layers defined in YAML
-   Paint/erase updates label arrays
-   Save creates a new version folder
-   Reload loads latest version automatically

## D. Model Comparison Slots

-   4 selectable model slots exist
-   Selecting a model loads masks as read-only overlays
-   Slots can be hidden

## E. Bookmarks

-   Bookmarks capture sample_id + z + t + optional note
-   Clicking bookmark restores correct location

## F. Run Agent (Stub)

-   Run Agent launches configured command
-   GUI reflects running state
-   Refresh discovers new model outputs

------------------------------------------------------------------------

# Architecture Rules

1.  Separate UI, state, I/O, and workers
2.  No heavy I/O in UI thread
3.  Enforce explicit axis handling
4.  Avoid redundant image loading
5.  Keep logic modular and testable

------------------------------------------------------------------------

# Versioning Rules

-   Masks saved to: `gt_root/versions/v0001/`
-   Each label layer saved as: `original_filename__{layer}.tif`
-   Each version folder includes `manifest.json`

------------------------------------------------------------------------

# Implementation Milestones

1.  YAML parsing + validation
2.  Dataset browser
3.  Z/T scroll enforcement
4.  Editable label layers
5.  Versioned mask saving/loading
6.  Comparison grid
7.  Bookmarks
8.  Agent execution stub

------------------------------------------------------------------------

# Completion Requirements

-   Installable via `pip install -e .`
-   README includes:
    -   YAML example
    -   Example project tree
    -   Quickstart instructions
-   Tests pass:
    -   YAML validation
    -   Version folder increment
    -   Mask roundtrip save/load

------------------------------------------------------------------------

End of AGENT_INSTRUCTIONS.md
