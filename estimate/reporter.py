# ========== reporter.py ==========

import numpy as np
import pandas as pd
from pathlib import Path
from evaluator import (
    load_ground_truth_boxes,
    get_pred_boxes,
    match_boxes,
    build_confusion_matrix,
    metrics_from_confusion_matrix,
    ALL_CLASSES,
)
from config import (
    LABEL_DIR,
    OUTPUT_CSV_YOLO,
    OUTPUT_CSV_RTDETR,
    OUTPUT_CSV_PIPELINE,
    OUTPUT_CSV_COMPARE,
)


def summarize(df, label):
    print(f"\n{'='*50}")
    print(f"  สรุป : {label}")
    print(f"{'='*50}")
    print(f"ภาพทั้งหมด : {len(df)} ภาพ")
    print(f"เวลาเฉลี่ย : {df['time_ms'].mean():.2f} ms/ภาพ")


def save_csv(df_yolo, df_rtdetr, df_pipeline):
    df_yolo.to_csv(OUTPUT_CSV_YOLO,         index=False)
    df_rtdetr.to_csv(OUTPUT_CSV_RTDETR,     index=False)
    df_pipeline.to_csv(OUTPUT_CSV_PIPELINE, index=False)


def _cm_to_dataframe(cm, classes):
    """แปลง numpy CM → DataFrame ที่อ่านง่าย (มี header background)"""
    labels = list(classes) + ["background"]
    df = pd.DataFrame(cm, index=labels, columns=labels)
    df.index.name   = "GT \\ Pred"
    return df


def evaluate_per_image(result_pairs, method_name, iou_threshold=0.5,
                       classes=None):
    """
    result_pairs = list of (img_name, result_object)

    เก็บค่าสำหรับ:
      1) per-image metrics (TP/FP/FN/P/R/F1 รวม)
      2) Confusion Matrix รวมทั้ง dataset (per-class + background)

    Return: df_per_image, summary_dict, confusion_matrix(np.array)
    """
    if classes is None:
        classes = ALL_CLASSES

    label_dir = Path(LABEL_DIR)
    rows      = []
    total_tp  = total_fp = total_fn = 0

    # CM รวมของทั้ง dataset
    cm_total = np.zeros((len(classes) + 1, len(classes) + 1), dtype=int)

    for img_name, result in result_pairs:
        txt_name = Path(img_name).stem + ".txt"
        gt_boxes = load_ground_truth_boxes(label_dir / txt_name)
        pb       = get_pred_boxes(result)

        # 1) Metric รวมแบบเดิม (สำหรับเทียบรวม)
        tp, fp, fn = match_boxes(gt_boxes, pb, iou_threshold)
        total_tp  += tp
        total_fp  += fp
        total_fn  += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0)

        # 2) สะสม Confusion Matrix
        cm_img = build_confusion_matrix(gt_boxes, pb, classes, iou_threshold)
        cm_total += cm_img

        rows.append({
            "image":      img_name,
            "gt_boxes":   len(gt_boxes),
            "pred_boxes": len(pb),
            "TP":         tp,
            "FP":         fp,
            "FN":         fn,
            "Precision":  round(precision, 4),
            "Recall":     round(recall,    4),
            "F1":         round(f1,        4),
        })

        # print(f"  {img_name} | GT:{len(gt_boxes)} Pred:{len(pb)}"
        #       f" TP:{tp} FP:{fp} FN:{fn}"
        #       f" P:{precision:.2f} R:{recall:.2f} F1:{f1:.2f}")

    # Overall metrics (แบบรวมเดิม — ตรวจ class ตอน matching)
    precision_all = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall_all    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_all        = (2 * precision_all * recall_all / (precision_all + recall_all)
                     if (precision_all + recall_all) > 0 else 0)

    # ===== สรุปจาก Confusion Matrix (per-class + macro/micro) =====
    cm_metrics = metrics_from_confusion_matrix(cm_total, classes)

    print(f"\n{'='*50}")
    print(f"  Overall : {method_name}")
    print(f"{'='*50}")
    print(f"  TP: {total_tp}  FP: {total_fp}  FN: {total_fn}")
    print(f"  Precision : {precision_all:.4f}")
    print(f"  Recall    : {recall_all:.4f}")
    print(f"  F1 Score  : {f1_all:.4f}")

    # แสดง Confusion Matrix
    print(f"\n  Confusion Matrix ({method_name}):")
    df_cm = _cm_to_dataframe(cm_total, classes)
    print(df_cm.to_string())

    print(f"\n  Per-Class Metrics:")
    for cls in classes:
        m = cm_metrics["per_class"][cls]
        print(f"    {cls:15s} TP:{m['TP']:3d} FP:{m['FP']:3d} FN:{m['FN']:3d}"
              f"  P:{m['Precision']:.4f} R:{m['Recall']:.4f} F1:{m['F1']:.4f}")
    print(f"    Macro Avg     P:{cm_metrics['macro']['Precision']:.4f}"
          f" R:{cm_metrics['macro']['Recall']:.4f}"
          f" F1:{cm_metrics['macro']['F1']:.4f}")

    df_eval = pd.DataFrame(rows)
    summary = {
        "Method":    method_name,
        "TP":        total_tp,
        "FP":        total_fp,
        "FN":        total_fn,
        "Precision": round(precision_all, 4),
        "Recall":    round(recall_all,    4),
        "F1":        round(f1_all,        4),
        # เพิ่ม metric per-class
        **{f"P_{c}":  cm_metrics["per_class"][c]["Precision"] for c in classes},
        **{f"R_{c}":  cm_metrics["per_class"][c]["Recall"]    for c in classes},
        **{f"F1_{c}": cm_metrics["per_class"][c]["F1"]        for c in classes},
        "Macro_P":   cm_metrics["macro"]["Precision"],
        "Macro_R":   cm_metrics["macro"]["Recall"],
        "Macro_F1":  cm_metrics["macro"]["F1"],
    }
    return df_eval, summary, cm_total


def save_confusion_matrix(cm, classes, method_name, out_path):
    """บันทึก Confusion Matrix เป็น CSV (อ่านง่าย มี header)"""
    df_cm = _cm_to_dataframe(cm, classes)
    df_cm.to_csv(out_path)
    print(f"  → {out_path}")


def compare_and_save(pairs_yolo, pairs_rtdetr, pairs_pipeline,
                     iou_threshold=0.5):
    classes = ALL_CLASSES

    df_eval_y, m_y, cm_y = evaluate_per_image(
        pairs_yolo,     "YOLO only",  iou_threshold, classes)
    df_eval_r, m_r, cm_r = evaluate_per_image(
        pairs_rtdetr,   "RT-DETR only", iou_threshold, classes)
    df_eval_p, m_p, cm_p = evaluate_per_image(
        pairs_pipeline, "Pipeline",   iou_threshold, classes)

    # บันทึก per-image
    df_eval_y.to_csv("eval_yolo.csv",     index=False)
    df_eval_r.to_csv("eval_rtdetr.csv",   index=False)
    df_eval_p.to_csv("eval_pipeline.csv", index=False)

    # บันทึก Confusion Matrix แต่ละแบบ
    print(f"\n{'='*50}")
    print("  บันทึก Confusion Matrix")
    print(f"{'='*50}")
    save_confusion_matrix(cm_y, classes, "YOLO only",   "cm_yolo.csv")
    save_confusion_matrix(cm_r, classes, "RT-DETR only","cm_rtdetr.csv")
    save_confusion_matrix(cm_p, classes, "Pipeline",    "cm_pipeline.csv")

    # ตารางเปรียบเทียบรวม
    df_compare = pd.DataFrame([m_y, m_r, m_p])
    df_compare.to_csv(OUTPUT_CSV_COMPARE, index=False)

    # คอลัมน์ที่จะแสดง (รวม per-class)
    show_cols = ["Method", "Precision", "Recall", "F1"]
    for c in classes:
        show_cols += [f"P_{c}", f"R_{c}", f"F1_{c}"]
    show_cols += ["Macro_P", "Macro_R", "Macro_F1"]

    print(f"\n{'='*50}")
    print("  ตารางเปรียบเทียบ 3 แบบ (รวม per-class)")
    print(f"{'='*50}")
    print(df_compare[show_cols].to_string(index=False))

    print(f"\nบันทึกไฟล์เสร็จแล้ว:")
    print(f"  → eval_yolo.csv / eval_rtdetr.csv / eval_pipeline.csv")
    print(f"  → cm_yolo.csv   / cm_rtdetr.csv   / cm_pipeline.csv")
    print(f"  → {OUTPUT_CSV_COMPARE}")