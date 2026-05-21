# ========== decision.py ==========

import numpy as np
from config import DAMAGE_CLASS


# def _area(box):
#     x1, y1, x2, y2 = box
#     return max(0, x2 - x1) * max(0, y2 - y1)


# def _intersection_area(boxA, boxB):
#     ix1 = max(boxA[0], boxB[0])
#     iy1 = max(boxA[1], boxB[1])
#     ix2 = min(boxA[2], boxB[2])
#     iy2 = min(boxA[3], boxB[3])
#     return max(0, ix2 - ix1) * max(0, iy2 - iy1)


def aggregate_overlap_ratio(damage_dets, normal_dets, frame_shape):
    """
    คำนวณสัดส่วน pixel ของ damage ที่ทับซ้อนกับพื้นที่ normal
    ใช้ numpy mask (vectorized) — เร็วกว่า pixel-loop เดิมหลายร้อยเท่า

    Args:
        damage_dets : list ของ detection ที่เป็น damage
        normal_dets : list ของ detection ที่เป็น normal
        frame_shape : tuple (h, w) หรือ (h, w, c) ของเฟรม

    Return:
        overlap ratio (0.0 - 1.0)
    """
    if not damage_dets or not normal_dets:
        return 0.0

    h, w = frame_shape[:2]

    # สร้าง mask 2 ตัว — boolean array ขนาดเท่าเฟรม
    dmg_mask = np.zeros((h, w), dtype=bool)
    nrm_mask = np.zeros((h, w), dtype=bool)

    # ระบาย damage box เป็น True (clip ค่าไม่ให้หลุดขอบภาพ)
    for det in damage_dets:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        dmg_mask[y1:y2, x1:x2] = True

    # ใช้ union bounding box ของ normal (ให้ตรงกับ logic เดิม)
    nx1 = min(d["xyxy"][0] for d in normal_dets)
    ny1 = min(d["xyxy"][1] for d in normal_dets)
    nx2 = max(d["xyxy"][2] for d in normal_dets)
    ny2 = max(d["xyxy"][3] for d in normal_dets)
    nx1, ny1, nx2, ny2 = int(nx1), int(ny1), int(nx2), int(ny2)
    nx1, nx2 = max(0, nx1), min(w, nx2)
    ny1, ny2 = max(0, ny1), min(h, ny2)
    nrm_mask[ny1:ny2, nx1:nx2] = True

    nrm_area = int(nrm_mask.sum())
    if nrm_area == 0:
        return 0.0

    # หา overlap = AND ของ 2 mask แล้วนับ pixel
    overlap_pixels = int((dmg_mask & nrm_mask).sum())

    return overlap_pixels / nrm_area


def has_damage(detections):
    return any(d["class_name"] == DAMAGE_CLASS for d in detections)


def pipeline_decision(stage_dets, frame_shape):
    """
    ตัดสินใจผลสุดท้าย
    Return: verdict, final_detections, overlap_pct
    """
    if has_damage(stage_dets):
        damage_dets = [d for d in stage_dets if "damage" in d["class_name"]]
        normal_dets = [d for d in stage_dets if "damage" not in d["class_name"]]
        overlap_pct = aggregate_overlap_ratio(damage_dets, normal_dets, frame_shape)
        return "DAMAGE", stage_dets, overlap_pct
    else:
        is_really_normal = any(d["class_name"] == "label_normal" for d in stage_dets)
        if is_really_normal:
            return "NORMAL", stage_dets, 0.0
   
    return "BACKGROUND", stage_dets, 0.0