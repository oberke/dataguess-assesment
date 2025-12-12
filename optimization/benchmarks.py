import time
import json
import torch
import numpy as np
import onnxruntime
import pynvml
import argparse
from ultralytics import YOLO

def get_gpu_utilization(handle):
    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu, mem_info.used / 1024**2 # GPU %, Mem Used MB
    except:
        return 0, 0

def benchmark_model(model_path, model_type='pytorch', img_size=640, iterations=100, warmup=10):
    print(f"Benchmarking {model_type} model: {model_path}")
    
    # Initialize PYNVML
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except:
        print("Warning: NVIDIA Driver not found, GPU metrics disabled.")
        handle = None

    # Load Model
    if model_type == 'pytorch':
        model = YOLO(model_path)
        dummy_input = np.random.rand(img_size, img_size, 3).astype(np.uint8) # YOLO handles preprocessing
    elif model_type == 'onnx':
        session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        # ONNX expects preprocessed input usually, but for fair comparison we simulate raw image passing if wrapper handles it, 
        # or pure inference time. Requirement says "pre/post-processing" should be measured or at least latency.
        # Here we verify pure inference latency mostly, but let's include basic resize as 'preprocess'
        dummy_input = np.random.rand(1, 3, img_size, img_size).astype(np.float32)

    latencies = []
    
    # Warmup
    print(f"Warming up for {warmup} iterations...")
    for _ in range(warmup):
        if model_type == 'pytorch':
            model(dummy_input, verbose=False)
        elif model_type == 'onnx':
            session.run(None, {input_name: dummy_input})

    # Benchmark
    print(f"Running {iterations} iterations...")
    start_time = time.time()
    
    gpu_usages = []
    
    for _ in range(iterations):
        iter_start = time.time()
        
        if model_type == 'pytorch':
            results = model(dummy_input, verbose=False)
        elif model_type == 'onnx':
            outputs = session.run(None, {input_name: dummy_input})
            
        iter_end = time.time()
        latencies.append((iter_end - iter_start) * 1000) # ms
        
        if handle:
            gpu_util, mem_used = get_gpu_utilization(handle)
            gpu_usages.append(gpu_util)
            
    total_time = time.time() - start_time
    
    # Metrics
    avg_latency = np.mean(latencies)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    fps = iterations / total_time
    avg_gpu = np.mean(gpu_usages) if gpu_usages else 0
    
    results = {
        "model": model_path,
        "type": model_type,
        "iterations": iterations,
        "latency_avg_ms": round(avg_latency, 2),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "fps": round(fps, 2),
        "gpu_util_avg_percent": round(avg_gpu, 2)
    }
    
    print("\nBenchmark Results:")
    print(json.dumps(results, indent=4))
    
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="../models/yolov8n.pt", help="Path to model file")
    parser.add_argument("--type", type=str, default="pytorch", choices=['pytorch', 'onnx', 'trt'], help="Model type")
    args = parser.parse_args()
    
    benchmark_model(args.model, args.type)
