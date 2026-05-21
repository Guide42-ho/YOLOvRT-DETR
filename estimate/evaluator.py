# ========== evaluator.py ==========

from pathlib import Path
import numpy as np
from config import DAMAGE_CLASS, NORMAL_CLASS, LABEL_DIR

CLASS_MAP = {
    0: DAMAGE_CLASS,
    1: NORMAL_CLASS,
}

# รายชื่อ class ทั้งหมดที่ใช้ในระบบ (ใช้สร้าง Confusion Matrix)
ALL_CLASSES = [DAMAGE_CLASS, NORMAL_CLASS]


def load_ground_truth_boxes(label_path):
    """
    อ่านไฟล์ .txt ของ YOLO
    Return: list of dict {class_name, cx, cy, w, h}
    """
    path = Path(label_path)
    if not path.exists():
        return []

    boxes = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            parts  = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append({
                "class_name": CLASS_MAP.get(cls_id, f"unknown_{cls_id}"),
                "cx": cx, "cy": cy, "w": w, "h": h,
            })
    return boxes


def yolo_to_xyxy(cx, cy, w, h):
    """แปลง YOLO format → x1y1x2y2"""
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


def calc_iou(box1, box2):
    """คำนวณ IoU ระหว่าง 2 กล่อง (YOLO format)"""
    x1a, y1a, x2a, y2a = yolo_to_xyxy(box1["cx"], box1["cy"], box1["w"], box1["h"])
    x1b, y1b, x2b, y2b = yolo_to_xyxy(box2["cx"], box2["cy"], box2["w"], box2["h"])

    # intersection
    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    # union
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0


def get_pred_boxes(result):
    """
    ดึง boxes จาก YOLO/RTDETR result
    Return: list of dict {class_name, cx, cy, w, h, conf}
    """
    boxes  = result[0].boxes
    names  = result[0].names
    img_h, img_w = result[0].orig_shape

    pred_boxes = []
    if boxes is None or len(boxes) == 0:
        return pred_boxes

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf   = float(boxes.conf[i].item())

        # xyxy → normalized cx cy w h
        x1, y1, x2, y2 = boxes.xyxy[i].tolist()
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = (x2 - x1) / img_w
        h  = (y2 - y1) / img_h

        pred_boxes.append({
            "class_name": names[cls_id],
            "cx": cx, "cy": cy, "w": w, "h": h,
            "conf": round(conf, 4),
        })

    return pred_boxes


def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    จับคู่ GT กับ Prediction โดยใช้ IoU (เฉพาะ class เดียวกัน)
    Return: TP, FP, FN per image
    """
    matched_gt   = set()
    matched_pred = set()
    tp = 0

    for pi, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gi  = -1

        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if pred["class_name"] != gt["class_name"]:
                continue

            iou = calc_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi

        if best_iou >= iou_threshold and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes)   - len(matched_gt)

    return tp, fp, fn


# ===================================================================
#                     Confusion Matrix (Object Detection)
# ===================================================================

def build_confusion_matrix(gt_boxes, pred_boxes, classes, iou_threshold=0.5):
    """
    สร้าง Confusion Matrix สำหรับ Object Detection ของ "1 ภาพ"

    Matrix ขนาด (N+1) x (N+1) โดย index N = "background"
        - แถว (row) = Ground Truth
        - หลัก (col) = Prediction

    หลักการ matching:
      1) คำนวณ IoU ทุกคู่ (gt, pred) — ไม่สนใจ class ก่อน
      2) Greedy match จาก IoU มากสุด → น้อยสุด ที่ ≥ threshold
         - ถ้า class ตรงกัน  → CM[gt_cls, pred_cls] += 1   (TP — diagonal)
         - ถ้า class ต่างกัน → CM[gt_cls, pred_cls] += 1   (misclassification — off-diagonal)
      3) pred ที่ไม่ถูก match  → CM[background, pred_cls] += 1   (FP)
      4) gt   ที่ไม่ถูก match  → CM[gt_cls, background]   += 1   (FN)
    """
    n = len(classes)
    cm = np.zeros((n + 1, n + 1), dtype=int)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    BG = n  # background index

    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return cm

    matched_gt   = set()
    matched_pred = set()

    # 1) เก็บ pair ที่ IoU ≥ threshold ทั้งหมด แล้ว sort จากมาก→น้อย
    if len(gt_boxes) > 0 and len(pred_boxes) > 0:
        pairs = []
        for gi in range(len(gt_boxes)):
            for pi in range(len(pred_boxes)):
                iou = calc_iou(gt_boxes[gi], pred_boxes[pi])
                if iou >= iou_threshold:
                    pairs.append((iou, gi, pi))
        pairs.sort(reverse=True)

        # 2) Greedy match
        for iou, gi, pi in pairs:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt.add(gi)
            matched_pred.add(pi)

            gt_cls   = gt_boxes[gi]["class_name"]
            pred_cls = pred_boxes[pi]["class_name"]
            if gt_cls in cls_to_idx and pred_cls in cls_to_idx:
                cm[cls_to_idx[gt_cls], cls_to_idx[pred_cls]] += 1

    # 3) FP: pred ที่ไม่ถูก match
    for pi, pr in enumerate(pred_boxes):
        if pi in matched_pred:
            continue
        pred_cls = pr["class_name"]
        if pred_cls in cls_to_idx:
            cm[BG, cls_to_idx[pred_cls]] += 1

    # 4) FN: gt ที่ไม่ถูก match
    for gi, gt in enumerate(gt_boxes):
        if gi in matched_gt:
            continue
        gt_cls = gt["class_name"]
        if gt_cls in cls_to_idx:
            cm[cls_to_idx[gt_cls], BG] += 1

    return cm


def metrics_from_confusion_matrix(cm, classes):
    """
    คำนวณ TP/FP/FN/Precision/Recall/F1 ต่อ class จาก Confusion Matrix

    cm shape = (N+1, N+1)  โดย index N = background

    - TP(c) = cm[c, c]
    - FP(c) = sum(cm[:, c]) - cm[c, c]   # ที่ทำนายเป็น c แต่จริงไม่ใช่ c (รวม background)
    - FN(c) = sum(cm[c, :]) - cm[c, c]   # ที่ GT เป็น c แต่ทำนายอื่น (รวม background)
    """
    n = len(classes)
    per_class = {}
    total_tp = total_fp = total_fn = 0

    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        per_class[cls] = {
            "TP": tp, "FP": fp, "FN": fn,
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            "F1":        round(f1, 4),
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Macro = ค่าเฉลี่ยของแต่ละ class
    macro_p = float(np.mean([per_class[c]["Precision"] for c in classes]))
    macro_r = float(np.mean([per_class[c]["Recall"]    for c in classes]))
    macro_f = float(np.mean([per_class[c]["F1"]        for c in classes]))

    # Micro = รวม TP/FP/FN ก่อนแล้วคำนวณ
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)
               if (micro_p + micro_r) > 0 else 0.0)

    return {
        "per_class": per_class,
        "macro": {
            "Precision": round(macro_p, 4),
            "Recall":    round(macro_r, 4),
            "F1":        round(macro_f, 4),
        },
        "micro": {
            "TP": total_tp, "FP": total_fp, "FN": total_fn,
            "Precision": round(micro_p, 4),
            "Recall":    round(micro_r, 4),
            "F1":        round(micro_f, 4),
        },
    }