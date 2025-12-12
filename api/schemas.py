from pydantic import BaseModel
from typing import List

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

class DetectionResponse(BaseModel):
    detections: List[BoundingBox]
    inference_time_ms: float
    backend: str

class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool

class Metrics(BaseModel):
    fps: float
    gpu_utilization: float
    latency_p50: float
    latency_p95: float
