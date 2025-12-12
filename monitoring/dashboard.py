import time
import collections
import json
import numpy as np
import threading
import os

try:
    import pynvml
except ImportError:
    pynvml = None

class PerformanceDashboard:
    def __init__(self, log_file="system_metrics.json", window_size=100):
        self.log_file = log_file
        self.window_size = window_size
        
        # Metrics Storage
        self.latency_buffer = collections.deque(maxlen=window_size)
        self.fps_buffer = collections.deque(maxlen=window_size)
        
        # GPU Init
        self.gpu_handle = None
        if pynvml:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except:
                print("Warning: GPU monitoring initialization failed.")
        
        # Ensure log file exists or clear it
        with open(self.log_file, "w") as f:
            f.write("")

    def update(self, latency_ms, fps_val):
        """
        Update metrics with a new observation
        """
        self.latency_buffer.append(latency_ms)
        self.fps_buffer.append(fps_val)
        
        # Collect snapshot
        metrics = self._collect_snapshot()
        self._log_to_json(metrics)
        return metrics

    def _collect_snapshot(self):
        # Latency Stats (P50, P90, P95)
        if self.latency_buffer:
            lats = list(self.latency_buffer)
            p50 = np.percentile(lats, 50)
            p90 = np.percentile(lats, 90)
            p95 = np.percentile(lats, 95)
            avg_lat = np.mean(lats)
        else:
            p50, p90, p95, avg_lat = 0, 0, 0, 0

        # GPU Stats
        gpu_util = 0
        mem_used = 0
        if self.gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util = util.gpu
                mem_used = mem.used / 1024**2 # MB
            except:
                pass

        return {
            "timestamp": time.time(),
            "latency": {
                "avg": round(avg_lat, 2),
                "p50": round(p50, 2),
                "p90": round(p90, 2),
                "p95": round(p95, 2)
            },
            "fps": round(self.fps_buffer[-1] if self.fps_buffer else 0, 2),
            "gpu": {
                "utilization_percent": gpu_util,
                "memory_used_mb": round(mem_used, 2)
            }
        }

    def _log_to_json(self, metrics):
        with open(self.log_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def print_summary(self, metrics):
        print(f"\rFPS: {metrics['fps']} | "
              f"Lat (P90): {metrics['latency']['p90']}ms | "
              f"GPU: {metrics['gpu']['utilization_percent']}%", end="")

if __name__ == "__main__":
    # Test
    dash = PerformanceDashboard()
    for i in range(200):
        lat = np.random.normal(30, 5) # Simulate ~30ms latency
        fps = 1000 / lat
        metrics = dash.update(lat, fps)
        dash.print_summary(metrics)
        time.sleep(0.05)
