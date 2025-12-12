
import cv2
import threading
import queue
import time
import numpy as np
import sys
import os

# Add project root to sys.path to allow imports from other modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector
from inference.tracker import Tracker
from inference.fusion import detect_drift
from monitoring.fps_meter import FPSMeter

class VideoEngine:
    def __init__(self, source, model_path, backend="pytorch"):
        self.cap = cv2.VideoCapture(source)
        self.detector = Detector(backend=backend, model_path=model_path)
        self.tracker = Tracker()
        self.fps_meter = FPSMeter()
        
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        self.stopped = False
        self.detection_interval = 3 # Run detector every N frames
        self.frame_count = 0
        # Persist tracks between detections
        self.current_tracks = []

    def start(self):
        # Start threads
        self.t_read = threading.Thread(target=self.read_frames)
        self.t_process = threading.Thread(target=self.process_frames)
        self.t_read.start()
        self.t_process.start()
        
        # Main thread handles visualization (OpenCV imshow needs main thread often)
        self.visualize()

    def read_frames(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            
            if not self.frame_queue.full():
                self.frame_queue.put(frame)
            else:
                time.sleep(0.01) # Small sleep to prevent tight loop

    def process_frames(self):
        while not self.stopped:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame = self.frame_queue.get()
            self.frame_count += 1
            
            # Logic: Detect every N frames, Track in between
            full_detection = (self.frame_count % self.detection_interval == 0)
            
            if full_detection:
                # Run Detector
                # expected output: list/tensor of [x1, y1, x2, y2, conf, cls]
                raw_dets = self.detector(frame) 
                
                # Convert raw_dets to numpy for tracker
                # This depends on Detector output format match
                if hasattr(raw_dets, 'xyxy'): # Ultralytics format
                    det_xyxy = raw_dets.xyxy.cpu().numpy()
                    det_conf = raw_dets.conf.cpu().numpy()
                    det_cls = raw_dets.cls.cpu().numpy()
                elif isinstance(raw_dets, np.ndarray) and raw_dets.shape[1] >= 6:
                     # Our Standardized Numpy Output [x1, y1, x2, y2, conf, cls]
                     det_xyxy = raw_dets[:, :4]
                     det_conf = raw_dets[:, 4]
                     det_cls = raw_dets[:, 5]
                else:
                    # Fallback/Mock
                    det_xyxy = np.empty((0, 4))
                    det_conf = np.empty((0,))
                    det_cls = np.empty((0,))
                
                # Update Tracker
                self.current_tracks = self.tracker.update(det_xyxy, det_conf, det_cls)
                
                # Log to console
                if len(det_cls) > 0:
                   print(f"Detected {len(det_cls)} objects.")

                # Drift Detection Check (Fusion)
                if detect_drift(det_xyxy, self.current_tracks[:, :4] if len(self.current_tracks) > 0 else []):
                    # print("Drift detected! Re-initializing tracker...")
                    self.tracker.reset()
                    self.current_tracks = self.tracker.update(det_xyxy, det_conf, det_cls)
            
            # Send (frame, tracks) to visualizer
            # Even if we didn't detect this frame, we send the last known tracks
            self.result_queue.put((frame, self.current_tracks))
            self.fps_meter.tick()

    def visualize(self):
        while not self.stopped:
            if self.result_queue.empty():
                time.sleep(0.01)
                continue
                
            frame, tracks = self.result_queue.get()
            
            # Draw tracks
            # tracks format: [x1, y1, x2, y2, track_id]
            if len(tracks) > 0: # Ensure tracks is not None/Empty
                for t in tracks:
                    x1, y1, x2, y2, tid = t
                    # Draw Bounding Box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Draw Label Background
                    label = f"ID: {int(tid)}"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame, (int(x1), int(y1)-20), (int(x1)+w, int(y1)), (0, 255, 0), -1)
                    # Draw Text
                    cv2.putText(frame, label, (int(x1), int(y1)-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            cv2.putText(frame, f"FPS: {self.fps_meter.get_fps():.2f}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("AI FAE Video Engine", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopped = True
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    import sys
    source = 0 
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        # Check if argument is a digit (webcam index)
        if arg.isdigit():
            source = int(arg)
        else:
            source = arg
    # Fix Model Path: Resolve relative to this script
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.pt")
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}. Trying fallback 'yolov8n.pt'")
        model_path = "yolov8n.pt"
        
    engine = VideoEngine(source, model_path)
    engine.start()
