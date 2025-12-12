# AI FAE Technical Assessment Project

## Overview
This repository contains a complete Edge AI Video Analytics System designed involves model training, optimization (TensorRT), multi-backend inference, and API deployment.

## Structure
- `training/`: Scripst for training YOLOv8 with advanced augmentations.
- `optimization/`: Pipeline to export to ONNX and build TensorRT engines (FP16/INT8).
- `inference/`: Real-time video engine with Detector, Tracker (ByteTrack), and periodic synchronization.
- `api/`: FastAPI application server with Docker support.
- `monitoring/`: System performance logging widgets.
- `tests/`: Unit tests for critical components.

## Setup
1. **Requirements**:
   - Python 3.8+
   - NVIDIA GPU + CUDA Toolkit (for TensorRT features)
   - Docker (for API deployment)

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: TensorRT python bindings usually require specific installation via NVIDIA index or container.*

## Usage

### 1. Training
```bash
cd training
python train.py
```
This will train YOLOv8n on the COCO8 dummy dataset and export an initial ONNX model.

### 2. Optimization
```bash
cd optimization
python export_to_onnx.py --weights ../training_logs/yolov8n_optim/weights/best.pt
python build_trt_engine.py --onnx ../models/model.onnx --precision fp16
```

### 3. Inference Demo (Webcam)
```bash
cd inference
python video_engine.py 0
```

### 4. API Service
Run locally:
```bash
cd api
uvicorn server:app --reload
```
Or with Docker:
```bash
cd api/docker
docker build -t ai-fae-app .
docker run --gpus all -p 8000:8000 ai-fae-app
```

## Monitoring & Testing
Run tests:
```bash
pytest tests/
```
