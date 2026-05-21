# ========== models.py ==========

from ultralytics import YOLO, RTDETR
from config import MODEL_YOLO_PATH, MODEL_RTDETR_PATH

def load_models():
    print("[โหลด] YOLOv8s ...")
    model_yolo   = YOLO(MODEL_YOLO_PATH)

    print("[โหลด] RT-DETR-L ...")
    model_rtdetr = RTDETR(MODEL_RTDETR_PATH)

    print(f"  YOLO   classes : {model_yolo.names}")
    print(f"  RTDETR classes : {model_rtdetr.names}")
    return model_yolo, model_rtdetr