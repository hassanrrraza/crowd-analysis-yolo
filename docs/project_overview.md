# Project overview: BRT crowd analysis (YOLOv8)

This document describes the **crowd-analysis-yolo** repository as implemented in the codebase: what it does, how the pieces fit together, and where to change behavior. It complements the root [README.md](../README.md), which covers installation, usage commands, and publication links.

## Purpose

The project is a **computer-vision pipeline** for monitoring crowds in **Peshawar Bus Rapid Transit (BRT)** footage. It uses **Ultralytics YOLOv8** to detect people, restricts counting to a **fixed polygon region of interest (ROI)** on each frame, and renders a **real-time dashboard** (counts, threshold bars, statistics, history graph, ROI overlay) with OpenCV.

**Research context:** The README cites an IEEE Xplore paper ([document 11204071](https://ieeexplore.ieee.org/document/11204071)); cite that source when using the work academically.

## Technology stack

| Area | Choice |
|------|--------|
| Runtime | Python 3.7+ (README) |
| Detection | `ultralytics` YOLO (`YOLO.predict`) |
| Video / UI | OpenCV (`cv2`), CVZone (`putTextRect`) |
| Numerics / tables | NumPy, pandas |
| Other deps | matplotlib, scikit-learn, Pillow, tqdm (see [requirements.txt](../requirements.txt)) |

Large artifacts (**`.pt`**, **video files**) are mostly **gitignored**; the repo may ship exception rules for specific weights (see [.gitignore](../.gitignore)).

## Repository layout (conceptual)

The tree below reflects what the code expects and what exists in a typical checkout. **Dataset folders** (`dataset/`, `images/`) hold many **YOLO-format label** files (`.txt`) and images; they are voluminous and are the training/evidence side of the project, not the minimal inference path.

```
crowd-analysis-yolo/
├── main.py                 # CLI entry: resolves paths, runs PeopleCounter
├── requirements.txt
├── README.md
├── coco1.txt               # Class names (root fallback)
├── best.pt, yolov8n.pt    # Weights (if present; may be gitignored except whitelisted)
├── data.txt                # Legacy YOLO data YAML fragment (paths point elsewhere)
├── data/                   # Expected home for coco1.txt, videos, images (varies by machine)
├── models/                 # Documented location for weights (optional)
├── src/
│   ├── people_counter.py # Core: inference loop + ROI counting + visualization
│   ├── get_coordinates.py # Helper: click frame to print coordinates (ROI design)
│   ├── utils/
│   │   └── image_extractor.py  # CLI: extract frames from video to disk
│   ├── models/__init__.py      # Placeholder package docstring only
│   └── __init__.py
├── notebooks/
│   └── yolov8_object_detection_on_custom_dataset.ipynb  # Training / tutorial workflow
├── yolov8_object_detection_on_custom_dataset.ipynb      # Duplicate notebook at repo root
├── dataset/                # Train/val images + labels (YOLO layout in practice)
├── images/                 # Additional images/labels
├── img.py                  # Small script: extract frames from abb.mp4 into dataset/
└── test.py                 # Standalone prototype / alternate ROI and resolution
```

## Execution flow

### 1. Command-line entry (`main.py`)

`main.py` parses arguments, **resolves file paths** in a predictable order (user path → `models/` or `data/` → script directory), builds `PeopleCounter`, and prints max/average counts after the run.

Supported arguments:

- `--model` (default `best.pt`)
- `--video` (default `cr.mp4`)
- `--classes` (default `coco1.txt`)
- `--threshold` (default `40`) — crowd alert reference count
- `--no-display` — headless processing (no OpenCV window)
- `--skip-frames` (default `3`) — process every *n*-th frame only

### 2. Core class (`src/people_counter.py`)

`PeopleCounter` encapsulates the full pipeline:

1. **Load model:** `self.model = YOLO(model_path)`.
2. **Load class names:** newline-separated labels from `class_file`. The shipped [data/coco1.txt](../data/coco1.txt) (and root `coco1.txt`) contain a **single class**, `person`, matching a single-class detector.
3. **Fixed ROI:** A quadrilateral `self.area1` is **hard-coded in pixel coordinates** for frames resized to **1280×720**. Only detections whose **bounding-box center** falls inside this polygon are counted (`cv2.pointPolygonTest` ≥ 0).
4. **Inference loop:**
   - Read video with `cv2.VideoCapture`.
   - Resize frame to 1280×720.
   - Every `skip_frames` frames, run `model.predict(frame)` and parse `results[0].boxes.data` via pandas.
   - For each row: box coords, class id `d`, map to `self.class_list[d]`, compute center, test against ROI, collect boxes for counting.
5. **Metrics:** Per processed frame, append `current_count` to `frame_counts`; track max count, rolling count history (last 100), smoothed FPS from inter-frame times.
6. **Visualization (if `display=True`):** Named window `Crowd Analysis`, draw ROI outline + light fill, detection boxes (thin rectangles with corner ticks), header, dual progress bars, statistics panel, mini line graph, footer (usage note + timestamp). **ESC** exits.

**Crowd status logic** (relative to `threshold`):

- **NORMAL:** count &lt; 60% of threshold  
- **WARNING:** 60%–90%  
- **CRITICAL:** ≥ 90%  

Colors are defined in `self.colors` (see source for BGR values). The README’s color table is the **intended UX**; compare with code if you need exact on-screen behavior.

### 3. Return value

`run()` returns a **Python list** of integer counts, one entry per **processed** frame (after skip-frame decimation), which `main.py` uses for simple max/average summaries.

## Supporting scripts

| File | Role |
|------|------|
| [src/utils/image_extractor.py](../src/utils/image_extractor.py) | `extract_frames()`: save up to *N* resized JPEGs from a video; CLI `--video`, `--output`, `--max-frames`, `--interval`, `--width`, `--height`. |
| [src/get_coordinates.py](../src/get_coordinates.py) | Plays `data/videos/cr.mp4` (or `cr.mp4` fallback) at 1280×720; **left-click** prints `(x, y)` for drafting ROI vertices; **n** advances; **ESC** quits. |
| [img.py](../img.py) | Extracts up to 100 frames from **`abb.mp4`** into `dataset/person_%d.jpg` (fixed resolution 1080×500). |
| [test.py](../test.py) | Early / alternate loop: different default polygon, **1020×500** resize, `cvzone.cornerRect`, mouse-move logging; useful as a reference but **not** wired to `main.py`. |

## Training and datasets

- **Notebook:** [notebooks/yolov8_object_detection_on_custom_dataset.ipynb](../notebooks/yolov8_object_detection_on_custom_dataset.ipynb) is a **Roboflow-style YOLOv8 tutorial** (install, CLI, pretrained inference, custom training). A **duplicate** exists at the repo root; prefer the `notebooks/` copy for organization.
- **data.txt:** Contains `train`, `val`, `nc`, and `names` entries pointing at paths like `../freedomtech/...` and a placeholder class name — treat as a **template or leftover** unless you repoint paths for your machine.
- **dataset/** and **images/**: Contain substantial **image and YOLO label** assets (`person_*.jpg` / `.txt`, `classes.txt`, train/val splits under `dataset/`). These support **custom training** and evaluation; they are not required to **run** `main.py` if you already have weights and a video.

## Configuration touchpoints

When adapting to a new scene, you will typically change:

1. **`self.area1`** in `PeopleCounter` (use `get_coordinates.py` to sample points on a 1280×720 frame).
2. **`threshold`** via CLI or constructor.
3. **`--skip-frames`** to trade accuracy for speed.
4. **Model path** (`--model` / `models/`).
5. **Class file** if your model has multiple classes (ensure line index `d` from YOLO matches `class_list`).

Keep **resize dimensions** and **ROI coordinates** in sync: counting assumes the same geometry used during ROI design.

## Testing and quality

There is **no automated test suite** in the repository; validation is **visual** (OpenCV window) and **quantitative** only via the printed max/average counts in `main.py`. For regression checks, compare counts on a short clip before and after changes, and verify ROI alignment on still frames.

## Licensing and data use

The README states **MIT**-style availability with attribution and notes that **BRT footage is for educational/research use** and must not be misused or shared inappropriately. Respect those constraints when handling video and derived outputs.

---

*Generated from the repository source tree and Python modules. Update this document if entry points, ROI defaults, or data layout change materially.*
