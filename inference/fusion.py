import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection + 1e-6)

def detect_drift(detector_boxes, tracker_boxes, threshold=0.5):
    """
    Compare detector outputs with tracker predictions to detect drift.
    Returns: bool (True if drift detected)
    """
    # Simple logic: If we have matched components but low IoU, it's a drift.
    # This is a heuristic. A robust implementation would use hungarian matching.
    
    if len(detector_boxes) == 0 or len(tracker_boxes) == 0:
        return False
        
    # Check max IoU for each tracker box against all detector boxes
    drifts = 0
    for t_box in tracker_boxes:
        max_iou = 0
        for d_box in detector_boxes:
            iou = calculate_iou(t_box, d_box)
            if iou > max_iou:
                max_iou = iou
        
        if max_iou < threshold:
            drifts += 1
            
    # If substantial portion of tracks are drifting
    if len(tracker_boxes) > 0 and (drifts / len(tracker_boxes)) > 0.5:
        return True
        
    return False
