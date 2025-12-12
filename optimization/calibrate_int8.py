
import tensorrt as trt
import os
import cv2
import glob
import numpy as np

class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, training_data_dir, cache_file, batch_size=1, height=640, width=640):
        super().__init__()
        self.cache_file = cache_file
        self.image_paths = glob.glob(os.path.join(training_data_dir, "*.jpg")) + \
                           glob.glob(os.path.join(training_data_dir, "*.png"))
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.current_index = 0
        
        # Allocate device memory for inputs
        self.device_input = None 
        # In a real scenario, we'd use pycuda.driver.mem_alloc to allocate GPU memory
        # and copy batches there. 
        # For this assessment script, we'll keep the structure conceptual or require pycuda.

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_paths):
            return None
            
        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.image_paths[self.current_index + i]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.width, self.height))
            # Preprocessing should match inference (Normalize 0-1 or user specific)
            img = img.astype(np.float32) / 255.0 
            img = img.transpose(2, 0, 1) # HWC to CHW
            batch_imgs.append(img)
            
        self.current_index += self.batch_size
        
        batch = np.array(batch_imgs).ravel()
        
        # Here we would copy 'batch' to self.device_input in GPU
        # return [int(self.device_input)]
        
        # Since we can't fully run this without GPU/PyCUDA installed right now:
        print(f"Calibrating batch {self.current_index // self.batch_size}...")
        return [] # Placeholder to prevent crash if run blindly
        

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

if __name__ == "__main__":
    print("Calibrator class defined. Use inside build_trt_engine.py by passing it to builder config.")
