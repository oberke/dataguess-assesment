from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
import uvicorn
import cv2
import numpy as np
import time
import torch

from api.schemas import DetectionResponse, BoundingBox, HealthCheck, Metrics
# Import Detector from inference module (requires setting python path or relative import)
# Assuming run from root
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.detector import Detector

# Global State
model = None
app_state = {
    "fps": 0.0,
    "latency_history": []
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model
    try:
        # Default to PyTorch for easy run, can be configured via env var
        backend = os.getenv("INFERENCE_BACKEND", "pytorch")
        model_path = os.getenv("MODEL_PATH", "../models/yolov8n.pt") 
        # Make sure path is absolute or correct relative
        if not os.path.exists(model_path):
             # Fallback to download or relative path check
             model_path = "yolov8n.pt" 
             
        model = Detector(backend=backend, model_path=model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
    yield
    # Cleanup
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await file.read()
    print(f"[DEBUG] Received file size: {len(contents)} bytes")
    if len(contents) > 0:
        print(f"[DEBUG] Header bytes: {contents[:10]}")

    nparr = np.frombuffer(contents, np.uint8)
    # Use IMREAD_UNCHANGED to correctly read alpha channels if present
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        print("[DEBUG] cv2.imdecode returned None. Trying PIL fallback for AVIF/HEIC...")
        try:
            from PIL import Image
            import io
            # Fallback: support AVIF/WEBP via Pillow (requires pillow-avif-plugin)
            image_pil = Image.open(io.BytesIO(contents))
            # Convert PIL RGB to OpenCV BGR
            img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            print("[DEBUG] PIL fallback successful.")
        except Exception as e:
            print(f"[DEBUG] PIL fallback failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid image data. Could not decode image.")

    # Convert to standard RGB/BGR if alpha channel exists (4 channels)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
    # Ensure it is 3 channel BGR for YOLO
    if len(img.shape) == 2: # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    start_time = time.time()
    detections = model(img) # Returns list of boxes (numpy array [N, 6] expected)
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000
    app_state["latency_history"].append(inference_time)
    if len(app_state["latency_history"]) > 100:
        app_state["latency_history"].pop(0)

    # Conversion logic for Standardized Numpy Output
    response_boxes = []
    
    # Check if we have detections (check if valid numpy array and not empty)
    if isinstance(detections, np.ndarray) and detections.size > 0:
        for box in detections:
            # box is a row: [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2 = box[:4].tolist()
            conf = float(box[4])
            cls_id = int(box[5])
            
            response_boxes.append(BoundingBox(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=conf,
                class_id=cls_id,
                class_name=str(cls_id)
            ))
            
    return DetectionResponse(
        detections=response_boxes,
        inference_time_ms=inference_time,
        backend=model.backend
    )

@app.get("/health", response_model=HealthCheck)
def health():
    return HealthCheck(
        status="ok",
        model_loaded=(model is not None),
        gpu_available=torch.cuda.is_available()
    )

@app.get("/metrics", response_model=Metrics)
def metrics():
    import collections
    lats = app_state["latency_history"]
    if not lats:
        p50, p95 = 0.0, 0.0
    else:
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        
    return Metrics(
        fps=1000.0 / p50 if p50 > 0 else 0.0,
        gpu_utilization=0.0, # Placeholder, requires nvml
        latency_p50=round(p50, 2),
        latency_p95=round(p95, 2)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
