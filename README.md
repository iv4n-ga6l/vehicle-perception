# Vehicle Perception

A real-time vehicle perception system that analyzes road traffic video input continuously using computer vision.

## Features

- **Lane Detection**: Detects road lane lines using edge detection and Hough transforms
- **Vehicle Detection**: 3D object detection using YOLO11 for cars, trucks, buses, and motorcycles
- **Bird's Eye View**: Perspective transformation with depth estimation

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run:
```bash
python main.py
```