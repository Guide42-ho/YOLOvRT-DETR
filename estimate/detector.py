# ========== detector.py ==========

from config import DEVICE, CONF_SOLO

# def get_classes(result):
#     """ดึงชื่อ class จาก result"""
#     boxes = result[0].boxes
#     names = result[0].names
#     detected = []
#     if boxes is not None and len(boxes) > 0:
#         for cls_id in boxes.cls.tolist():
#             detected.append(names[int(cls_id)])
#     return detected

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



def detect_yolo(model, frame):
    """รัน YOLOv8s และคืน classes"""
    result = model(frame, device=DEVICE, conf=0.45, verbose=False )
    return result


def detect_rtdetr(model, frame):
    """รัน RT-DETR และคืน classes"""
    result = model(frame, device=DEVICE, conf=0.64, verbose=False )
    return result
    