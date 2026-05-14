# Custom YOLOv8 training and deployment

This guide is for readers who want to **train a model on their own images**, save weights as **`best.pt`** (or any **`.pt`** file), and **run inference on their own videos** with this project’s crowd-analysis app.

- **Stack:** [Ultralytics YOLOv8](https://docs.ultralytics.com/) (same family as this repo’s `ultralytics` dependency).
- **Where to run:** **Jupyter** on your PC, or **Google Colab** (GPU runtime recommended).
- **Deeper setup** (venv, `main.py` paths): [detailed_setup_guide.md](detailed_setup_guide.md).
- **How inference works in this repo:** [project_overview.md](project_overview.md).
- **Full notebook code:** For the complete, cell-by-cell tutorial (install, CLI examples, Colab/Drive, training flow), open [yolov8_object_detection_on_custom_dataset.ipynb](../notebooks/yolov8_object_detection_on_custom_dataset.ipynb) or the equivalent at the repo root: [yolov8_object_detection_on_custom_dataset.ipynb](../yolov8_object_detection_on_custom_dataset.ipynb) (see §8).

---

## 1. What you will produce

After training, Ultralytics typically writes:

| File | Typical location |
|------|-------------------|
| **`best.pt`** | `runs/detect/train*/weights/best.pt` (best validation metric during training) |
| **`last.pt`** | same folder — weights from the **last** epoch |

You can **rename** weights (e.g. `my_crowd.pt`). The crowd app loads any compatible **YOLOv8 detection** checkpoint via `--model path\to\weights.pt`.

---

## 2. Dataset: images + YOLO labels

### 2.1 Layout

Use **object detection** with one annotation file per image (**YOLO `.txt`** format):

```text
your_dataset/
  images/
    train/          # .jpg / .png …
    val/
  labels/
    train/          # same stem as image, e.g. img_001.txt
    val/
```

Each **`labels/.../*.txt`** line is:

```text
class_id center_x center_y width height
```

All five numbers are **normalized** to **0–1** relative to image width/height (`class_id` is integer starting at **0**).

### 2.2 Dataset YAML (`data.yaml`)

Create a small config next to your data (paths can be absolute or relative to this file):

```yaml
path: /absolute/or/relative/path/to/your_dataset   # optional root
train: images/train
val: images/val

names:
  0: person
```

- **Single class (recommended for this BRT counter):** one name, id `0`, e.g. `person`. This matches the shipped [data/coco1.txt](../data/coco1.txt) idea (one line per class).
- **Multiple classes:** list every class in **the same order** as training (`0`, `1`, …). You must then update the **class names file** this app uses (see §7).

**Practical tip:** Export **“YOLOv8”** from [Roboflow](https://roboflow.com/) or similar tools to obtain `data.yaml`, images, and labels without hand-editing.

---

## 3. Environment (Jupyter or Colab)

### 3.1 Install Ultralytics

In a notebook cell:

```python
%pip install -U ultralytics
```

Restart the kernel if the installer asks you to. Optionally pin a version for reproducibility (the tutorial notebook in this repo once used `8.0.20`; newer `8.x` is usually fine—match versions across train and deploy if you hit odd errors).

```python
from ultralytics import YOLO
```

### 3.2 Google Colab

1. **Runtime → Change runtime type → GPU** (e.g. T4).
2. Upload your dataset ZIP or **mount Google Drive** and unzip to `/content` (or similar).
3. Point `data.yaml`’s `path` / `train` / `val` at that folder.

Minimal Drive pattern:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Then unzip and set YAML paths to the extracted root.

### 3.3 Local Jupyter

Use a **GPU** machine if available; CPU training is possible but slow. Keep `data.yaml` paths valid on your filesystem (Windows: prefer forward slashes or raw strings in Python).

---

## 4. Training

### 4.1 Python API (works well in notebooks)

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # or yolov8s.pt / m / l / x for larger models
model.train(
    data="path/to/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,          # lower if you run out of VRAM
    project="runs/detect",
    name="my_train",
)
```

Weights appear under:

```text
runs/detect/my_train/weights/best.pt
runs/detect/my_train/weights/last.pt
```

### 4.2 CLI (optional)

```bash
yolo task=detect mode=train model=yolov8n.pt data=path/to/data.yaml epochs=100 imgsz=640
```

See [Ultralytics train docs](https://docs.ultralytics.com/modes/train/) for augmentations, resume, and multi-GPU flags.

### 4.3 Quick sanity check before the full app

```python
from ultralytics import YOLO
model = YOLO("runs/detect/my_train/weights/best.pt")
model.predict(source="path/to/test_video.mp4", save=True)
```

Preview clips or saved frames under Ultralytics’ `runs/` output folders.

---

## 5. Copy weights to this project

1. Copy **`best.pt`** (or your chosen `.pt`) to the repo root, `models/`, or any folder you prefer.
2. Run the crowd app with an explicit path if needed:

```bash
python main.py --model models/best.pt --video path\to\your_video.mp4
```

Path resolution for `--model` is described in [detailed_setup_guide.md](detailed_setup_guide.md).

---

## 6. Class names file (`coco1.txt`)

`PeopleCounter` maps each detection’s **class index** `d` to **line `d`** in the class file (`--classes`, default `coco1.txt`).

- **Single-class `person` model:** file content is exactly:

```text
person
```

- **Multi-class model:** add **one line per class** in **YOLO id order** (line 0 = class 0, line 1 = class 1, …). Example for `person` and `bag`:

```text
person
bag
```

**Important:** The current counting logic adds **every** detection whose box center falls inside the ROI, regardless of class. For production crowd counting, train a **person-only** model or **extend the code** to filter `d == person_class_id` if you use multiple classes.

---

## 7. Region of interest (ROI) and video geometry

This app resizes frames to **1280×720** and counts inside a **fixed polygon** in `src/people_counter.py`. Your **new video** may not match the old scene.

1. Run `python src/get_coordinates.py` on a representative clip (same resize as production).
2. Paste coordinates into `self.area1` in `people_counter.py`.

Otherwise counts will be wrong even with a perfect model.

---

## 8. In-repo notebook (full code)

**Check out this notebook for the full runnable workflow**—every install line, `yolo` command, and training step is in the cells:

- **Preferred path:** [notebooks/yolov8_object_detection_on_custom_dataset.ipynb](../notebooks/yolov8_object_detection_on_custom_dataset.ipynb)
- **Same tutorial may also exist at repo root:** [yolov8_object_detection_on_custom_dataset.ipynb](../yolov8_object_detection_on_custom_dataset.ipynb)

The two copies can differ over time; use whichever you keep updated. The notebook is tutorial-style (sample zips, Colab paths, pinned `ultralytics` versions)—adapt paths and package versions to **your** dataset and environment.

---

## 9. Checklist

| Step | Done |
|------|------|
| Images + YOLO labels, train/val split | |
| `data.yaml` with correct `names` and paths | |
| Train in Jupyter/Colab; obtain `best.pt` | |
| `predict` on a short test video | |
| Copy `.pt` into project; `python main.py --model ... --video ...` | |
| `coco1.txt` lines match model class order | |
| ROI updated for your camera/layout | |

---

## 10. References

- Ultralytics YOLOv8: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- Academic citation for this BRT project: see [README.md](../README.md) (IEEE Xplore link).

When you ship a real-world deployment, version-control **which `best.pt`**, **which YAML**, and **which ROI** you used together so results stay reproducible.
