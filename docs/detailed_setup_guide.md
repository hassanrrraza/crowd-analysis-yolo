# Detailed setup guide

This guide walks you from a clean machine to a working **crowd-analysis-yolo** run. For architecture and module behavior, see [project_overview.md](project_overview.md). For a shorter quick start, see the root [README.md](../README.md).

---

## 1. What you are setting up

After setup you will be able to:

- Run **`python main.py`** to process a video with YOLOv8, count people inside a fixed on-screen region, and view the OpenCV dashboard (or run headless).
- Optionally use **`src/utils/image_extractor.py`** to dump frames for labeling, and **`src/get_coordinates.py`** to click pixel coordinates when tuning the ROI.

**Ethics / data:** BRT footage in this project is intended for **education and research** only. Do not redistribute video in ways that violate rights or misrepresent the scene. See the [README](../README.md).

---

## 2. Requirements

### 2.1 Software

| Item | Notes |
|------|--------|
| **Python** | **3.7+** (README). A current 64-bit Python **3.10–3.12** is a practical choice on Windows. |
| **Git** | Optional, for `git clone`; you can also use a ZIP download. |
| **`pip`** | Usually bundled with Python. |

### 2.2 Hardware

| Item | Notes |
|------|--------|
| **CPU** | Sufficient to run YOLO on CPU; expect **lower FPS** than on GPU. |
| **GPU (optional)** | **CUDA-capable NVIDIA GPU** recommended for near–real-time playback. Install a **CUDA-enabled PyTorch** build if you use GPU (see §5.2). |
| **RAM** | **8 GB+** recommended; more helps with large videos and background apps. |
| **Display** | Required for the default UI (`display=True`). Use `--no-display` on servers without a GUI (§8.3). |

---

## 3. Get the repository

### 3.1 Clone with Git

```bash
git clone https://github.com/hassanrrraza/crowd-analysis-yolo.git
cd crowd-analysis-yolo
```

### 3.2 Or unpack a ZIP

Extract the archive so the project root contains `main.py`, `src/`, `requirements.txt`, etc. All commands below assume your shell’s **current directory is that project root**.

**Windows tip:** In PowerShell, `cd` to the folder that contains `main.py`:

```powershell
cd C:\Users\YourName\crowd-analysis-yolo
```

---

## 4. Create a virtual environment (recommended)

Isolating dependencies avoids conflicts with other Python projects.

### 4.1 Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 4.2 Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Keep the venv **activated** for the install and run steps below.

---

## 5. Install Python dependencies

### 5.1 Install from `requirements.txt`

From the **project root** (where `requirements.txt` lives):

```bash
pip install -r requirements.txt
```

This pulls in at minimum:

- `opencv-python`, `numpy`, `pandas`
- `ultralytics` (YOLOv8)
- `cvzone`
- `torch` (see §5.2)
- `matplotlib`, `scikit-learn`, `Pillow`, `tqdm`

### 5.2 PyTorch and GPU (optional but important for speed)

`requirements.txt` pins `torch>=1.9.0`, which **pip** may install as a **CPU** wheel. For **NVIDIA GPU**:

1. Open [PyTorch — Get Started](https://pytorch.org/get-started/locally/).
2. Choose your OS, package **pip**, CUDA version that matches your driver.
3. Run the **exact** `pip install torch ...` command PyTorch shows (often with `--index-url` for CUDA wheels).
4. You can install PyTorch **first**, then run `pip install -r requirements.txt` again if needed (pip will usually leave an already-satisfied `torch` alone, or upgrade—read pip’s output).

**Verify CUDA (optional):**

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

`True` and a GPU name means Ultralytics can use the GPU when you run inference.

### 5.3 Smoke test imports

```bash
python -c "import cv2, torch, ultralytics; from src.people_counter import PeopleCounter; print('ok')"
```

If this prints `ok`, core packages and the package layout resolve correctly from the project root.

---

## 6. Files you need before the first run

### 6.1 Model weights (`.pt`)

The app loads a YOLO model via Ultralytics:

- Default CLI flag: `--model best.pt`

**Path resolution** in `main.py` (in order):

1. The path you pass, if it exists.
2. `models/<filename>` (e.g. `models/best.pt`).
3. `<project_root>/<filename>`.

**Repository note:** `.gitignore` excludes many `*.pt` files but may **whitelist** specific names (e.g. `best.pt`, `yolov8n.pt`). Your checkout might already include weights at the **repo root**. If `best.pt` is missing, add your trained or downloaded weights under `models/` or the root and pass `--model path\to\weights.pt`.

The bundled workflow assumes a **single-class “person”** model consistent with `coco1.txt` (§6.3).

### 6.2 Input video

- Default CLI flag: `--video cr.mp4`

**Path resolution** in `main.py` (in order):

1. The path you pass, if it exists.
2. `data/videos/<filename>` (e.g. `data/videos/cr.mp4`).
3. `<project_root>/<filename>`.

**Create the folder if needed:**

```text
data/videos/
```

Place your video there, or pass an **absolute path**:

```bash
python main.py --video D:\Videos\my_clip.mp4
```

### 6.3 Class names file

- Default: `--classes coco1.txt`

**Path resolution:**

1. As given, if it exists.
2. `data/coco1.txt`.
3. `<project_root>/coco1.txt`.

**Format:** One class name per line, no extra spaces. The training artifact in-repo uses a **single** class; [data/coco1.txt](../data/coco1.txt) contains:

```text
person
```

The detector output uses class index `d` as a **line index** into this file. If you switch to a multi-class model, you must align **model class order** with this file or update the file accordingly.

---

## 7. Run the application

### 7.1 Basic run

From **project root**, with venv activated:

```bash
python main.py
```

The program prints which **model**, **video**, and **class file** paths it resolved, then opens a window titled **`Crowd Analysis`** (unless `--no-display`). Press **ESC** in that window to stop.

### 7.2 Command-line reference

| Argument | Default | Meaning |
|----------|---------|---------|
| `--model` | `best.pt` | YOLO weights path (see §6.1). |
| `--video` | `cr.mp4` | Video path (see §6.2). |
| `--classes` | `coco1.txt` | Class names file (see §6.3). |
| `--threshold` | `40` | Reference count for status bars (NORMAL / WARNING / CRITICAL bands in code). |
| `--no-display` | off | Process without showing a window (still reads video and prints summary stats). |
| `--skip-frames` | `3` | Process **every Nth** frame (`1` = every frame; higher = faster, jerkier). |

Examples:

```bash
python main.py --threshold 50 --skip-frames 2
python main.py --model models\best.pt --video data\videos\cr.mp4 --classes data\coco1.txt
python main.py --help
```

### 7.3 Where counting happens (ROI)

`PeopleCounter` resizes frames to **1280×720** and counts detections whose **bounding-box center** lies inside a **hard-coded polygon** in `src/people_counter.py`. If your video resolution or scene differs, counts will be wrong until you **edit that polygon** using coordinates from **`src/get_coordinates.py`** (§9.2).

---

## 8. Use the Python API

From the project root (or with `PYTHONPATH` set to the root), you can call the same class the CLI uses:

```python
from src.people_counter import PeopleCounter

counter = PeopleCounter(
    model_path="models/best.pt",          # or "best.pt", or absolute path
    video_path="data/videos/cr.mp4",
    class_file="data/coco1.txt",
    threshold=40,
)

counts = counter.run(display=True, skip_frames=3)
print("max", max(counts) if counts else 0)
```

- `run(..., display=False)` matches `--no-display`.
- Return value: **list of per-processed-frame** integer counts (only frames after skip decimation).

---

## 9. Optional tools after setup

### 9.1 Extract frames for labeling

```bash
python src/utils/image_extractor.py --video data/videos/cr.mp4 --output data/images/extracted --max-frames 200
```

Useful flags: `--interval`, `--width`, `--height` (defaults resize before saving). The script opens a small preview window; **ESC** stops early.

### 9.2 Pick ROI coordinates (`get_coordinates.py`)

This script plays **`data/videos/cr.mp4`** (or falls back to **`cr.mp4`** at the root) at **1280×720**.

- **Left-click:** prints `Coordinate: (x, y)` to the terminal; use those points as polygon vertices in `people_counter.py`.
- **`n`:** the script’s on-screen instructions mention moving to the **next frame** (see `src/get_coordinates.py`); playback is a tight read/display loop—if you need true frame-by-frame stepping, pause by adding a higher `waitKey` delay or temporarily changing the loop.
- **ESC:** exit.

Run:

```bash
python src/get_coordinates.py
```

Ensure the same video and resize convention as production, or coordinates will not match.

### 9.3 Other scripts (legacy)

- **`img.py`** — extracts frames from **`abb.mp4`** into `dataset/`; hard-coded filenames and resolution. Useful as a snippet, not the main pipeline.
- **`test.py`** — standalone prototype with a **different** default ROI and frame size; not invoked by `main.py`.

---

## 10. Training and notebooks

Custom training is documented in the Jupyter notebook:

- Prefer: `notebooks/yolov8_object_detection_on_custom_dataset.ipynb`
- A duplicate may exist at the repo root; content is tutorial-style (Ultralytics install, CLI, training).

**Setup:**

```bash
pip install jupyter
jupyter notebook
```

Open the notebook and follow its cells. The root **`data.txt`** file is a **YAML-style fragment** with paths pointing at external folders; treat it as a **template** and update `train:` / `val:` to your machine before training.

---

## 11. Troubleshooting

### 11.1 `ModuleNotFoundError`

- Activate the **same** venv you used for `pip install`.
- Run commands from **project root** so `src.people_counter` resolves.
- Re-run `pip install -r requirements.txt`.

### 11.2 Video opens but `ret` is false / “Using video” path wrong

- Confirm the printed path exists (typos, wrong drive letter on Windows).
- Try an absolute `--video` path.
- Re-encode the file (H.264 in `.mp4`) if OpenCV’s backend fails on a rare codec.

### 11.3 Window does not appear / hang on server

- SSH or Docker often has **no display**. Use `python main.py --no-display` or `counter.run(display=False)`.
- On Linux headless setups you may need a virtual framebuffer (**Xvfb**) only if you insist on GUI code paths—prefer `--no-display`.

### 11.4 Wrong counts or always zero

- **ROI mismatch:** Polygon in `people_counter.py` is for **1280×720**; re-measure with `get_coordinates.py` on the same resize.
- **Model / class mismatch:** Single-class `coco1.txt` must match model output indices.
- **`skip-frames` too high:** Skipping too many frames can miss short appearances (does not usually zero out all counts unless nothing enters the ROI).

### 11.5 CUDA out of memory

- Use a smaller YOLO model, reduce resolution in code (advanced), or run on CPU (slower).
- Close other GPU applications.

### 11.6 Git LFS / large files

Weights and videos may be **gitignored** or stored via **Git LFS**. If something is missing after clone, check the README, releases, or obtain files separately (§6.1–6.2).

---

## 12. Checklist (minimal path to success)

1. [ ] Python 3.7+ installed; `python --version` works.
2. [ ] Repo cloned or extracted; shell `cd` to project root.
3. [ ] Virtualenv created and **activated**.
4. [ ] `pip install -r requirements.txt` completed without errors.
5. [ ] (Optional) CUDA PyTorch installed and `torch.cuda.is_available()` is `True`.
6. [ ] `best.pt` (or your weights) reachable via `--model` / `models/`.
7. [ ] Video reachable via `--video` / `data/videos/`.
8. [ ] `coco1.txt` present and aligned with model classes.
9. [ ] `python main.py` runs; ESC stops the GUI run.
10. [ ] If counts look wrong, update ROI using `get_coordinates.py` and edit `src/people_counter.py`.

---

## 13. Citation

For academic use, cite the IEEE Xplore paper linked in the [README](../README.md) ([document 11204071](https://ieeexplore.ieee.org/document/11204071)).

---

*This guide reflects the repository’s `main.py`, `src/people_counter.py`, `requirements.txt`, and path-fallback behavior as of the documented layout. If you change defaults in code, update this document or add a note in your fork.*
