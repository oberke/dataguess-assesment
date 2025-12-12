from supervision import ByteTrack, Detections
import numpy as np

class Tracker:
    def __init__(self):
        self.tracker = ByteTrack()
    
    def update(self, detections_xyxy, confidences, class_ids):
        """
        Update tracker with new detections.
        detections_xyxy: np.array of shape (N, 4)
        confidences: np.array of shape (N,)
        class_ids: np.array of shape (N,)
        """
        if len(detections_xyxy) == 0:
            return np.empty((0, 5)) # xyxy + track_id
            
        detections = Detections(
            xyxy=detections_xyxy,
            confidence=confidences,
            class_id=class_ids
        )
        
        # Supervision's ByteTrack update returns Detections object with tracker_id set
        tracks = self.tracker.update_with_detections(detections)
        
        # Return format: [x1, y1, x2, y2, track_id]
        if len(tracks.xyxy) > 0:
            return np.hstack((tracks.xyxy, tracks.tracker_id.reshape(-1, 1)))
        else:
            return np.empty((0, 5))

    def reset(self):
        self.tracker.reset()
