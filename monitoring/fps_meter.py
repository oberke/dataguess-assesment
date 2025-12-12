import time
import collections

class FPSMeter:
    def __init__(self, buffer_len=100):
        self.timestamps = collections.deque(maxlen=buffer_len)
        
    def tick(self):
        self.timestamps.append(time.time())
        
    def get_fps(self):
        if len(self.timestamps) < 2:
            return 0.0
        
        duration = self.timestamps[-1] - self.timestamps[0]
        if duration <= 0:
            return 0.0
            
        return (len(self.timestamps) - 1) / duration
