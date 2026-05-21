# ========== runner.py ==========

import time
from detector import  detect_rtdetr, detect_yolo, get_detections
from decision import has_damage, pipeline_decision


def run_yolo_only(model, frame, img_name=""):
    """แบบที่ 1 — YOLO อย่างเดียว"""
    
    t_start = time.time()
    result  = detect_yolo(model,frame) 
    elapsed = (time.time() - t_start) * 1000

    classes = get_detections(result)
    classes = [d["class_name"] for d in classes]    
    classes_str = ", ".join(classes) if classes else "background"

    log = {
        "image":         img_name,
        "classes":       classes_str,
        "time_ms":       round(elapsed, 2),
    }
    return log, result          # ← คืน result ด้วย


def run_rtdetr_only(model, frame, img_name=""):
    """แบบที่ 2 — RT-DETR อย่างเดียว"""
    
    t_start = time.time()
    result  = detect_rtdetr(model, frame)
    elapsed = (time.time() - t_start) * 1000
    
    classes = get_detections(result)
    classes = [d["class_name"] for d in classes]
    classes_str = ", ".join(classes) if classes else "background"
    
    log = {
        "image":         img_name,
        "classes":       classes_str,
        "time_ms":       round(elapsed, 2),
    }
    return log, result          # ← คืน result ด้วย


def run_pipeline(model_yolo, model_rtdetr, frame, img_name=""):
    """แบบที่ 3 — Pipeline (YOLO กรอง + RT-DETR ยืนยัน)"""
    
    t_start = time.time()

    # Stage 1 — YOLO กรอง
    result_s1   = detect_yolo(model_yolo, frame)
    stage2_used  = False
    result_s2    = []
    classes_str1 = get_detections(result_s1)

    # Stage 2 — RT-DETR ยืนยัน (เฉพาะถ้าเจอ damage)
    if has_damage(classes_str1):
        stage2_used  = True
        result_s2    = detect_rtdetr(model_rtdetr, frame) 
    
    result_final = result_s2 if stage2_used else result_s1
    
    elapsed = (time.time() - t_start) * 1000
    
    
    classes_str2 = get_detections(result_s2) if stage2_used else []
    classes = [d["class_name"] for d in classes_str1]
    classes_str1 = ", ".join(classes) if classes else "background"
    classes = [d["class_name"] for d in classes_str2]
    classes_str2 = ", ".join(classes) if classes else "background"

    log = {
        "image":          img_name,
        "stage1_classes": classes_str1,
        "stage2_used":    stage2_used,
        "stage2_classes": classes_str2,
        "time_ms":        round(elapsed, 2),
    }
    return log, result_final