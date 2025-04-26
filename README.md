
# 🌎 Object Detection & Tracking Benchmark

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)  
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%7C%20Ultralytics%20%7C%20YOLOX-red)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

Welcome to the **Object Detection & Tracking Benchmark** project!  
This repository evaluates **15 state-of-the-art object detection models** across different frameworks — integrating **ByteTrack** for robust tracking and real-time **object counting**.

Tested on Google Colab, the project focuses on:
- **Inference Speed (FPS)**
- **Detection Efficiency**
- **Real-time Tracking Accuracy**

---

## 🔍 Key Features

✅ Multi-framework model support (YOLOv5, YOLOv8, YOLOX, Faster R-CNN, EfficientDet)  
✅ High-performance object tracking using **ByteTrack**  
✅ Automated object counting with line-crossing detection  
✅ Optimized for **real-time inference** (CUDA acceleration)  
✅ Customizable video processing (frame skipping, adaptive thresholds, resizing)  
✅ Modular, scalable, and easy-to-extend design  

---

## 📈 Key Findings

| Category | Best Models | Notes |
|:---|:---|:---|
| ⚡ Fastest (Real-Time) | YOLOv8s (59.53 FPS), YOLOv5su (58.66 FPS), YOLO11s (57.61 FPS), YOLOX-s (56.2 FPS) | Ideal for high-speed applications |
| 🎯 Most Accurate (but Slower) | Faster R-CNN (10.6 FPS), EfficientDet D3 (10.59 FPS) | Best suited for offline, high-accuracy tasks |
| ⚖️ Best Balance (Speed vs Performance) | YOLO12s (41.42 FPS), YOLOX-m (39.92 FPS), YOLOv5mu (43.69 FPS) | Good compromise for general-purpose detection |

*Note: YOLO12l (21.35 FPS) offers higher precision but lower speed.*

---

## 🏗️ Project Structure

- **Model Management**: Unified loading for YOLO, YOLOX, Faster R-CNN, and EfficientDet models.
- **Detection Pipelines**: Modular detectors for each model family.
- **Tracking**: Integrated with **ByteTrack** for robust object ID tracking.
- **Counting System**: Counts objects crossing user-defined lines.
- **Summary Export**: FPS, detection stats, and counting results exported to `.txt` and `.csv`.
- **Memory Management**: Automatic GPU memory cleaning during long processing sessions.
- **Display Manager**: Real-time visualization with FPS, tracking IDs, and class labels.

---

## 🚀 Installation

```bash
# Install required packages
pip install torch torchvision effdet deep-sort-realtime ultralytics tqdm opencv-python

# (Optional) Clone and install YOLOX
git clone https://github.com/georaiser/YOLOX.git
cd YOLOX
pip install -v -e .
cd ..
```

---

## ⚙️ Usage

```python
# Example for running a model
from ObjectDetectionBenchmark import MODELManager

model_handler = MODELManager("yolov8s")
model = model_handler.get_model()
# Follow detection -> tracking -> counting pipeline
```

You can easily extend or swap models in `ModelConfig` or `DetectorManager`.

---

## 💡 Conclusion

- For **real-time tracking**: Choose **YOLOv8s**, **YOLOv5su**, or **YOLO11s**.  
- For **accuracy-focused tasks**: Prefer **Faster R-CNN** or **EfficientDet D3**.  
- For **balanced tasks**: Use **YOLO12s** or **YOLOX-m**.

📌 **This benchmark serves as a roadmap for selecting the best model based on your application's needs.**

---

## 📫 Let's Connect!

Have questions, ideas, or want to suggest improvements?  
**Open an issue or start a discussion!**

---

# 📄 License

This project is licensed under the [MIT License](LICENSE).
