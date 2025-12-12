
import logging
import json
import time

class SystemLogger:
    def __init__(self, log_file="system_metrics.json"):
        self.log_file = log_file
        # Setup standard python logger if needed, or custom JSON logger
    
    def log_metric(self, metric_name, value):
        entry = {
            "timestamp": time.time(),
            "metric": metric_name,
            "value": value
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
