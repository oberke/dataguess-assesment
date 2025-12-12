import cv2
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    coords[:, 0].clip(0, img0_shape[1], out=coords[:, 0])  # x1
    coords[:, 1].clip(0, img0_shape[0], out=coords[:, 1])  # y1
    coords[:, 2].clip(0, img0_shape[1], out=coords[:, 2])  # x2
    coords[:, 3].clip(0, img0_shape[0], out=coords[:, 3])  # y2
    return coords

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None):
    """
    Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # Select 0th output

    # Checks
    if isinstance(prediction, np.ndarray):
        pass # Numpy is good
    else:
        # Convert torch to numpy if needed
        prediction = prediction.cpu().numpy()

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 4  # number of classes
    
    output = [np.zeros((0, 6))] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        # Transpose if [4+nc, 8400] -> [8400, 4+nc]
        x = x.transpose() 
        
        # Constraints
        # (x[:, 4:] is class scores)
        # Find max confidence
        scores = x[:, 4:]
        conf = scores.max(axis=1)
        j = scores.argmax(axis=1)
        
        # Filter by confidence
        mask = conf > conf_thres
        x = x[mask]
        conf = conf[mask]
        j = j[mask]
        
        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        # We need to construct this
        
        # NMS
        # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h] (top-left) usually, or we can use our xyxy
        # Let's use cv2.dnn.NMSBoxes which is robust
        
        # Convert xyxy back to xywh top-left for cv2 NMS
        # or just implement simple nms here? cv2 is faster.
        
        boxes_for_cv2 = np.copy(box)
        boxes_for_cv2[:, 2] = boxes_for_cv2[:, 2] - boxes_for_cv2[:, 0] # w
        boxes_for_cv2[:, 3] = boxes_for_cv2[:, 3] - boxes_for_cv2[:, 1] # h
        
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_for_cv2.tolist(),
            scores=conf.tolist(),
            score_threshold=conf_thres,
            nms_threshold=iou_thres
        )
        
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            # Append to output: x1,y1,x2,y2, conf, cls
            dets = np.concatenate((box[indices], conf[indices][:, None], j[indices][:, None].astype(np.float32)), axis=1)
            output[xi] = dets
            
    return output
