# TR_Park

**TR_Park** is a computer vision project developed for the SIH Hackathon. The project focuses on vehicle detection and traffic analysis using deep learning models and real-world traffic images and videos. It leverages state-of-the-art object detection algorithms (YOLOv3 and YOLOv8) to identify and count vehicles in various scenes, aiming to support smart city and traffic management solutions.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Sample Results](#sample-results)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Features

- 🚗 **Vehicle Detection:** Detects and counts cars and other vehicles in images and videos using YOLO models.
- 🎥 **Video and Image Processing:** Supports both still images and video streams for analysis.
- 📊 **Traffic Analysis:** Enables basic traffic density estimation for smart parking and urban planning applications.
- ⚡ **Real-Time Inference:** Designed for efficient, real-time detection on standard hardware.

---

## Project Structure

```
TR_Park/
├── .idea/                   # IDE configuration files
├── 1_car.jpg                # Sample image with a single car
├── 1car.jpeg                # Alternate sample image
├── Car_detection.py         # Vehicle detection script (YOLOv3)
├── coco.names               # Class labels for COCO dataset
├── multi_car.jpg            # Sample image with multiple cars
├── out.jpg                  # Output image with detection results
├── result.jpeg              # Output/result image
├── result.jpg               # Output/result image
├── traffic.jpg              # Sample traffic image
├── traffic2.jpg             # Additional sample image
├── traffic3.jpg             # Additional sample image
├── traffic_viedo3.mp4       # Sample traffic video
├── vechile_detection_v8.py  # Vehicle detection script (YOLOv8)
├── yolov3.cfg               # YOLOv3 model configuration
├── yolov3.weights           # YOLOv3 pre-trained weights
├── yolov8m.pt               # YOLOv8 model weights
└── README.md                # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/) (for YOLOv8)
- Other dependencies as required (see below)

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dasoam/TR_Park.git
   cd TR_Park
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy torch
   ```
   *You may need to install additional packages depending on your environment and model requirements.*

3. **Download YOLO weights (if not present):**
   - `yolov3.weights` and `yolov8m.pt` should be present in the repo. If not, download from the official sources.

---

## Usage

### For YOLOv3 (Image Detection)

```bash
python Car_detection.py --image traffic.jpg
```
- This will process `traffic.jpg` and output the detection results (e.g., `out.jpg`).

### For YOLOv8 (Image Detection)

```bash
python vechile_detection_v8.py --image traffic2.jpg
```
- Replace the image filename as needed.

### For Video Detection

- Modify the script to use `traffic_viedo3.mp4` as input, or adapt for webcam/stream input.

---

## Model Details

- **YOLOv3:** Uses `yolov3.cfg` and `yolov3.weights` for object detection. Classes are defined in `coco.names`.
- **YOLOv8:** Uses `yolov8m.pt` for improved detection accuracy and speed.

---

## Sample Results

Sample input and output images are included in the repository:
- `1_car.jpg`, `multi_car.jpg`, `traffic.jpg`, etc.
- Output images: `out.jpg`, `result.jpg`, `result.jpeg`

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request.

---

## Contact

- **Author:** Dasoam
- **Repository:** [github.com/Dasoam/TR_Park](https://github.com/Dasoam/TR_Park)
- For questions or support, please open an issue in the repository.

---
