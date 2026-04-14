# BRT Crowd Analysis System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

### Publication

This repository accompanies peer-reviewed work on IEEE Xplore:

- **Research paper:** [IEEE Xplore document 11204071](https://ieeexplore.ieee.org/document/11204071)
- **Author profile:** [Hassan Raza on IEEE Xplore](https://ieeexplore.ieee.org/author/710100950356345)

If you use this project in academic or technical publications, please cite the paper linked above.

---

## 📋 Overview

An advanced crowd analysis system for Peshawar's Bus Rapid Transit (BRT) using YOLOv8 and computer vision techniques. The system performs real-time crowd detection and monitoring to improve safety and efficiency in BRT stations.

- **Accuracy**: 98% in detecting and counting people within specified regions of interest
- **Purpose**: Enables proactive crowd management by providing accurate monitoring data
- **Application**: Optimizes resource allocation and prevents overcrowding situations

> **Note:** The BRT video footage used in this project is strictly for educational and research purposes only. The footage should not be shared online or used in any misleading way. All rights to the footage belong to their respective owners.

## ✨ Key Features

- **Real-time Detection & Counting**
  - Accurate people detection using YOLOv8
  - Custom region of interest (ROI) definition
  - Threshold-based crowd monitoring

- **Advanced Visualization Interface**
  - Modern UI with professional header/footer
  - Color-coded status indicators (Normal/Warning/Critical)
  - Comprehensive statistics panel
  - Real-time count history graph
  - Semi-transparent ROI visualization
  - Elegant detection boxes with corner highlights

- **Developer Tools**
  - Frame extraction utility for dataset creation
  - Support for custom trained models
  - Detailed performance metrics

## 🖼️ Visualization Interface

<p align="center">
  <img src="https://github.com/hassanrrraza/crowd-analysis-yolo/raw/main/data/images/interface_preview.PNG" width="700" alt="Interface Preview">
</p>

The system features a modern and informative visualization interface designed for effective crowd monitoring:

### Display Elements

| Component | Description |
|-----------|-------------|
| **Header Bar** | Professional title display with clean, modern design |
| **Counting Information** | Real-time count with color-coded progress bars |
| **Statistics Panel** | Shows current/max/avg counts, FPS, and runtime |
| **Count History Graph** | Real-time mini-graph showing count trends |
| **Detection Visualization** | Elegant bounding boxes with corner highlights |
| **Footer Information** | Timestamp and copyright information |

### Color Scheme

| Status | Color Code | Usage |
|--------|------------|-------|
| Normal | Green (#00FF00) | Below 60% of threshold |
| Warning | Orange (#FFA500) | Between 60-90% of threshold |
| Critical | Red (#FF0000) | Above 90% of threshold |
| Accent | Gold (#FFCC00) | UI highlights and borders |

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended for real-time processing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/hassanrrraza/crowd-analysis-yolo.git
   cd crowd-analysis-yolo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights** (if not included)
   - Place YOLOv8 weights files in the `models/` directory
   - Or use the included `best.pt` custom trained model

## 💻 Usage

### Command Line Interface

```bash
# Basic usage
python main.py

# With custom parameters
python main.py --model best.pt --video cr.mp4 --threshold 50

# Show all available options
python main.py --help
```

### Python API

```python
from src.people_counter import PeopleCounter

# Initialize the counter
counter = PeopleCounter(
    model_path="models/best.pt",
    video_path="data/videos/cr.mp4",
    class_file="data/coco1.txt",
    threshold=40
)

# Run with visualization
counter.run(display=True)
```

### Extracting Frames

```bash
python src/utils/image_extractor.py --video data/videos/cr.mp4 --output data/images/extracted --max-frames 200
```

## 📁 Project Structure

```
crowd-analysis-yolo/
├── data/                  # Data files
│   ├── videos/            # Video files for analysis
│   ├── images/            # Extracted frames and training images
│   ├── labels/            # Annotation files for training
│   └── coco1.txt          # Class names file
├── models/                # Trained YOLO model files
├── notebooks/             # Jupyter notebooks for experimentation
├── src/                   # Source code
│   ├── people_counter.py  # Main people counting module
│   ├── utils/             # Utility functions
│   │   └── image_extractor.py  # Frame extraction utility
│   └── models/            # Model loading and management
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```

## 🔧 Training Custom Models

For custom model training, refer to the Jupyter notebook `notebooks/yolov8_object_detection_on_custom_dataset.ipynb` which provides step-by-step instructions for:

1. Preparing your dataset
2. Configuring training parameters
3. Training the model
4. Evaluating performance
5. Exporting for inference

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [CVZone](https://github.com/cvzone/cvzone)
- [OpenCV](https://opencv.org/)

## 👤 Author

<img src="https://github.com/hassanrrraza.png" width="100px" style="border-radius:50%">

**Hassan Raza**
- Email: [hassan2056764@gmail.com](mailto:hassan2056764@gmail.com)
- LinkedIn: [hassanrrraza](https://www.linkedin.com/in/hassanrrraza/)
- GitHub: [hassanrrraza](https://github.com/hassanrrraza)
- IEEE Xplore: [Author profile](https://ieeexplore.ieee.org/author/710100950356345)

## 📄 License

© 2026 Hassan Raza. All rights reserved.
This project is available for use under the MIT license. You are free to use, modify, and distribute this code in your work, provided that you give appropriate credit to the original author.

For formal citation, use the IEEE Xplore publication: [https://ieeexplore.ieee.org/document/11204071](https://ieeexplore.ieee.org/document/11204071).