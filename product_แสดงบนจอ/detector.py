# ========== detector.py ==========

from config import DEVICE, CONF_STAGE1, CONF_STAGE2

def get_detections(result):
    """
    ดึง boxes พร้อม class และ confidence
    Return: list of dict {class_name, conf, xyxy}
    """
    boxes  = result[0].boxes
    names  = result[0].names
    dets   = []

    if boxes is None or len(boxes) == 0:
        return dets

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())
        xyxy   = boxes.xyxy[i].tolist()
        dets.append({
            "class_name": names[cls_id],
            "conf":       round(conf, 2),
            "xyxy":       xyxy,       # [x1, y1, x2, y2]
        })
    return dets


def detect_stage1(model, frame):
    result = model(frame, device=DEVICE, conf=CONF_STAGE1, verbose=False)
    return get_detections(result)


def detect_stage2(model, frame):
    result = model(frame, device=DEVICE, conf=CONF_STAGE2, verbose=False)
    return get_detections(result)