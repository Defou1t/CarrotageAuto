# CarrotageAuto

**Automated digitization of Soviet-era well-log charts.** Turns scanned paper well logs
(raster JPEG/TIFF) into digital curves as NeuraLOG `.nlgx` working files (+ `.bck`) and LAS,
ready for expert QC — with as little manual correction as possible.

> Internal development log (detailed, Russian): [`PLAN_new_dialog.md`](PLAN_new_dialog.md).
> This README is the outward-facing overview for users and developers.

---

## The problem

Old Soviet geophysical well logs ("каротаж") are hand-drawn paper charts: multiple curves per
track, a depth grid, scale rulers, handwritten numbers, and **back-up / amplified scales**
(the same sonde re-plotted at ×5 / ×25 / ×125 when it runs off-scale). They were scanned to
raster images. We convert them back into calibrated digital curves.

The hard parts are **identity** (which ink belongs to which curve, especially where curves cross
or share a colour) and **back-up scales** (1× and 5× of one sonde are *separate lines with a gap*,
not one continuous wrap). A naïve "binary mask + centroid" tracer loses both.

## Approach

Two pipelines live in this repo:

| | Legacy (U-Net) | **Current (image-understanding)** |
|---|---|---|
| Foreground | trained U-Net stroke mask | colour classification + geometric masks |
| Separation | geometric multi-line tracker | per-colour instance extraction + identity linker |
| Training | yes (`unet_best.pt`) | **none — rule-based** |
| Runtime | Python + PyTorch (GPU) | pure Python 3.14 + OpenCV (CPU) |
| Status | superseded (binary mask loses identity) | active |

The current pipeline is **identity-aware and image-understanding-first**, organised in phases:

- **Phase A — frame & scales** (foundation)
  - `A1` read the scale ruler with a vision LLM → which scales exist and their ranges
  - `A2/A2b` rebuild the depth grid at the native step, snapped to the real ink lines
  - `A3` mask non-curve zones (numbers, ticks, grid cells, ruler band) so they are never traced
- **Phase B — identity & behaviour** (the quality layer)
  - `B1` extract line **instances** from the image (centreline, start/end, thickness, **colour**)
  - `B2` behavioural priors — SP is smooth, resistivity/caliper are peaky (measured over the corpus)
  - `B3` identity tracker — links strokes into lines by colour + trajectory; a line never jumps
    onto a neighbour. **Swap rate ≈ 0 %** on validation, including same-colour pairs.
- **Phase C — values** (in progress): map traced shapes to real values across back-up scales.

Validation is always against expert ground-truth (`.nlgx` traces): per-curve pixel error,
coverage, and **swap rate** (does a line stay on its own identity).

## Quick start

The whole pipeline has one entry point — `digitizer/pipeline.py`:

```bash
# Digitize one file or a whole folder of scans → *_auto.nlgx (+ .bck)
python digitizer/pipeline.py process <file.nlgx | folder> [--patch-scan] [--regrid]

# Add verified nlgx+las to the learning corpus and refresh priors.json
python digitizer/pipeline.py learn <folder-of-verified-nlgx+las>
```

`process` needs a per-scan NeuraLOG **template** `.nlgx` (calibration + curve definitions) next to
the image; it produces the trace. Output opens directly in NeuraLOG for QC over the scan.
`process` is pure OpenCV/NumPy — **no model file, no GPU required**.

## Repository structure

```
CarrotageAuto/
├─ README.md                 ← you are here
├─ PLAN_new_dialog.md        internal dev log / roadmap (Russian, authoritative status)
├─ neuralog_context.md       NeuraLOG UI-automation context (legacy path)
├─ mnemonics.json            curve dictionary (names, units, groups) from the client
├─ *.py (repo root)          legacy NeuraLOG UI automation (Win32 clicks) — superseded
└─ digitizer/                STANDALONE digitizer (the real product)
```

### `digitizer/` — module map (grouped by role)

**Core nlgx I/O (foundation, no internal deps):**
- `extract_nlgx.py` — parse `.nlgx` (multi-IFD TIFF) → calibration, scale families, pixel traces
- `write_nlgx.py` — write `.nlgx` (round-trip + inject a trace into tag 35490) + `.bck`
- `dataset.py` / `dataset_build.py` — curve curation, mask rasterization, LAS matching, manifest

**Current pipeline (Phase A/B + end-to-end):**
- `detect_scales.py` — A1: ruler crop → VLM (constrained JSON) → scale structure
- `set_depth_grid.py` + `detect_calibration.py` — A2/A2b: depth grid rebuild + snap to ink
- `detect_masks.py` — A3: exclude-mask of non-curve zones
- `behavior_priors.py` — B2: smoothness/peakiness priors per curve class
- `extract_instances.py` — B1: colour → connected components → line instances + properties
- `track_identity.py` — B3: stroke linking into identity-stable lines
- `digitize_b3.py` — end-to-end B1→B3 → `_auto.nlgx (+bck)`
- **`pipeline.py`** — single entry point: `process` + `learn`

**Legacy U-Net path (superseded, kept for reference):**
- `unet.py`, `train.py`, `infer.py` — U-Net stroke segmentation
- `track_multi.py` — geometric multi-line tracker over the binary mask
- `digitize.py`, `cache_tracks.py`, `inject_trace.py`, `batch_digitize.py`

**Back-up scales / values & diagnostics:**
- `decode_levels.py` — decode amplified/back-up scale levels (DP/Viterbi)
- `survey_levels.py`, `eval_swaps.py`, `overlay_diff.py`, `batch_*` — analysis & QC tooling

Dependency roots: everything builds on `extract_nlgx`; `pipeline → digitize_b3 →
{track_identity → extract_instances → detect_masks, behavior_priors}`.

## Environments

| Task | Interpreter |
|---|---|
| New pipeline, nlgx I/O, analysis | **Python 3.14** + OpenCV/NumPy/Pillow (no SciPy, no Torch) |
| Legacy U-Net train/infer | **venv ComfyUI** — PyTorch 2.x + CUDA (RTX 5080) |
| Scale ruler reading (A1) | **LM Studio** local server, vision model `google/gemma-4-26b-a4b` |

## Where is the trained model?

The legacy U-Net checkpoint is at:

```
F:\nds\output\unet_data\unet_best.pt
```

(trained by `digitizer/train.py`; tiles live under `F:\nds\output\unet_data\`).
**The current pipeline does not use any trained model** — it is rule-based (colour + geometry +
behaviour), so there is no weights file to ship for it. "Learning" for the current pipeline means
growing the verified corpus and refreshing `priors.json` via `pipeline.py learn`.

## Data layout (per NeuraLOG convention)

```
projects/<well>/
├─ img/   raster scans (.jpg/.tif)
├─ wlg/   NeuraLOG working files (.nlgx + .bck)
└─ las/   exported digital data (.las)
```

## Domain background (NeuraLOG terms)

- **Depth Axis (DA)** — depth↔pixel calibration of a track. **Scale Axis (SA)** — value↔pixel.
- **Depth grids** — horizontal rules at even intervals (1:200 → every 4 m, 1:500 → every 10 m,
  "every 10 cells on the bold lines").
- **Back-up / amplified scales** — when a curve runs off-scale it continues on a back-up:
  *Backup Right/Left* (appears on the opposite side) or *amplified* (×5/×25…). In `.nlgx` these are
  separate Scale Axes chained by the `next` tag; the trace carries a per-depth *level* segment.
  Scanning at **200 dpi** is recommended (300 dpi only for 1-inch logs).
- Delivery = `.nlgx + .bck (+ .las)` per scan; the expert re-checks every curve in NeuraLOG, so the
  digitizer must *understand* the sheet (scales, identity, behaviour) rather than trace blindly.

## Status

Phase A complete; Phase B core complete (identity tracker, ~0 % swaps); end-to-end assembly and a
single-entry CLI in place. Active work: back-up-scale transitions (Phase C), coverage on faint
amplified lines, and line→curve mapping without a template for fully new scans. See
[`PLAN_new_dialog.md`](PLAN_new_dialog.md) for the live roadmap.
