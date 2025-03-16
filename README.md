## Author

- **Hassan Raza**
- Email: hassan2056764@gmail.com
- LinkedIn: [https://www.linkedin.com/in/hassanrrraza/](https://www.linkedin.com/in/hassanrrraza/)
- GitHub: [https://github.com/hassanrrraza](https://github.com/hassanrrraza)

# Crowd Analysis using YOLOv8

A computer vision project for crowd analysis and people counting using YOLOv8 object detection.

> **Note:** The BRT video footage used in this project is strictly for educational and research purposes only. The footage should not be shared online or used in any misleading way. All rights to the footage belong to their respective owners.


## Overview

This project implements an advanced crowd analysis system for Peshawar's Bus Rapid Transit (BRT) using YOLOv8 and computer vision techniques. The system performs real-time crowd detection and monitoring to improve safety and efficiency in BRT stations. Using custom-trained models on BRT station imagery, it achieves 98% accuracy in detecting and counting people within specified regions of interest. The solution enables proactive crowd management by providing accurate monitoring data to optimize resource allocation and prevent overcrowding situations.

## Features

- Real-time people detection and counting using YOLOv8
- Custom area definition for zone-based counting
- Advanced visualization interface:
  - Modern UI with professional header and footer
  - Real-time counting bar with color-coded status indicators
  - Dynamic threshold monitoring with percentage display
  - Comprehensive statistics panel showing:
    - Current count
    - Maximum count
    - Average count
    - FPS monitoring
    - Runtime tracking
  - Interactive mini-graph displaying count history
  - Semi-transparent ROI overlay for clear zone visualization
  - Elegant detection boxes with corner highlights
  - Professional timestamp and copyright information
- Frame extraction utility for dataset creation
- Support for custom trained models

## Project Structure

```
crowd-analysis-yolo/
├── data/
│   ├── videos/         # Video files for analysis
│   ├── images/         # Extracted frames and training images
│   ├── labels/         # Annotation files for training
│   └── coco1.txt       # Class names file
├── models/             # Trained YOLO model files
├── notebooks/          # Jupyter notebooks for experimentation
├── src/
│   ├── people_counter.py  # Main people counting module
│   ├── utils/
│   │   └── image_extractor.py   # Utility for extracting frames
│   └── models/         # Model loading and management
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/hassanrrraza/crowd-analysis-yolo.git
   cd crowd-analysis-yolo
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the model weights (if not already included):
   - Place YOLOv8 weights files in the `models/` directory
   - Or use the included `best.pt` custom trained model

## Usage

### Running People Counter

```python
from src.people_counter import PeopleCounter

# Initialize the counter with model and video paths
counter = PeopleCounter(
    model_path="models/best.pt",
    video_path="data/videos/cr.mp4",
    class_file="data/coco1.txt",
    threshold=40
)

# Run the counter with visualization
counter.run(display=True)
```

### Command Line Interface

```bash
# Basic usage
python main.py

# With custom parameters
python main.py --model best.pt --video cr.mp4 --threshold 50

# Additional options
python main.py --help  # Show all available options
```

The visualization interface will automatically adjust to your screen resolution while maintaining the 16:9 aspect ratio for optimal viewing.

### Extracting Frames from Video

```bash
python src/utils/image_extractor.py --video data/videos/cr.mp4 --output data/images/extracted --max-frames 200
```

## Training Custom Models

For custom model training, refer to the Jupyter notebook `notebooks/yolov8_object_detection_on_custom_dataset.ipynb` which provides step-by-step instructions.

## Requirements

- Python 3.7+
- OpenCV
- Pandas
- NumPy
- Ultralytics YOLOv5/YOLOv8
- CVZone

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [CVZone](https://github.com/cvzone/cvzone) 

## Copyright

© 2025 Hassan Raza. All rights reserved.

This project, "People Counter using YOLOv8", was developed and implemented by Hassan Raza. You are free to use, modify, and distribute this code in your work, provided that you give appropriate credit to the original author. This project demonstrates the practical application of computer vision and deep learning techniques for real-time people counting and tracking.

If you use this project in your work, please cite it as: "People Counter using YOLOv8 by Hassan Raza". For any questions or collaborations, feel free to reach out to the author.

## Visualization Interface

The system features a modern and informative visualization interface designed for effective crowd monitoring:

### Main Display Elements
1. **Header Bar**
   - Professional title display
   - Clean, modern design with accent highlights

2. **Counting Information**
   - Real-time count display with progress bar
   - Dynamic threshold percentage indicator
   - Color-coded status indicators (green for normal, orange for warning, red for critical)

3. **Statistics Panel**
   - Current and maximum crowd counts
   - Running average count
   - Real-time FPS monitoring
   - System runtime tracking
   - Clean, semi-transparent background for better visibility

4. **Count History Graph**
   - Real-time mini-graph showing count trends
   - Grid lines for better readability
   - Automatic scaling based on maximum counts

5. **Detection Visualization**
   - Elegant bounding boxes with corner highlights
   - Semi-transparent ROI overlay
   - Clear zone demarcation

6. **Footer Information**
   - Current date and time
   - Copyright information
   - Professional layout with consistent styling

### Color Scheme
- Normal Status: Green (#00FF00)
- Warning Status: Orange (#FFA500)
- Critical Status: Red (#FF0000)
- Accent Color: Gold (#FFCC00)
- Text Background: Dark Gray (#2C2C2C)
- Border Elements: Light Gray (#B4B4B4)