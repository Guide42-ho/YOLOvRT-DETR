# ========== decision.py ==========

from config import DAMAGE_CLASS, NORMAL_CLASS

# def verdict_from_classes(classes):
#     """ตัดสินใจจาก class ที่เจอ"""
#     if DAMAGE_CLASS in classes:
#         return "DAMAGED"
#     elif NORMAL_CLASS in classes:
#         return "NORMAL"
#     else:
#         return "(background)"

def has_damage(detections):
    return any(d["class_name"] == DAMAGE_CLASS for d in detections)

# def pipeline_decision(stage1_classes, stage2_classes, stage2_used):
#     """
#     Logic Pipeline
#     - damage มีความสำคัญกว่า normal เสมอ
#     - ถ้า RT-DETR ไม่ยืนยัน → ถือว่า normal
#     """
#     has_damage = DAMAGE_CLASS in stage1_classes

#     if not has_damage:
#         return verdict_from_classes(stage1_classes)

#     if stage2_used:
#         if DAMAGE_CLASS in stage2_classes:
#             return "DAMAGED"
#         else:
#             return "NORMAL (RT-DETR Not confirmed)"

#     return "No effect Stage2"

def pipeline_decision(stage1_dets, stage2_dets, stage2_used):
    """
    ตัดสินใจผลสุดท้าย
    Return: verdict, final_detections
    """
    if stage2_used:
        if has_damage(stage2_dets):
            # damage_dets = [d for d in stage2_dets if "damage" in d["class_name"]]
            # normal_dets = [d for d in stage2_dets if "damage" not in d["class_name"]]
            # overlap_pct = aggregate_overlap_ratio(damage_dets, normal_dets)     
            return "DAMAGED", stage2_dets, #overlap_pct
        else:
            return "NORMAL", stage2_dets, #0.0
        
    if not has_damage(stage1_dets):
        is_really_normal = any(d["class_name"] == "label_normal" for d in stage1_dets)
        if is_really_normal:
            return "NORMAL", stage1_dets,# 0.0
    
    return "UNKNOWN", stage1_dets,  #0.0
    