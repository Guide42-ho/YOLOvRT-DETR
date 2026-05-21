import cv2
import pandas as pd
from pathlib import Path

from models   import load_models
from runner   import run_yolo_only, run_rtdetr_only, run_pipeline
from reporter import summarize, save_csv, compare_and_save
from config   import TEST_DIR


def main():
    print("=" * 50)
    print("  Bottle Inspection — เปรียบเทียบ 3 แบบ")
    print("=" * 50)

    model_yolo, model_rtdetr = load_models()
    images = sorted(Path(TEST_DIR).glob("*.jpg"))
    print(f"\nพบภาพทั้งหมด: {len(images)} ภาพ\n")

    log_yolo     = []
    log_rtdetr   = []
    log_pipeline = []

    pairs_yolo     = []   # (img_name, result) สำหรับ evaluate
    pairs_rtdetr   = []
    pairs_pipeline = []

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠️ อ่านภาพไม่ได้: {img_path.name}")
            continue

        # print(f"\n📷 {img_path.name}")
        # print("-" * 50)

        # รัน 3 แบบ — แต่ละแบบคืน (log, result)
        log_y, res_y = run_yolo_only(model_yolo,  frame, img_path.name)
        log_r, res_r = run_rtdetr_only(model_rtdetr, frame, img_path.name)
        log_p, res_p = run_pipeline(model_yolo, model_rtdetr, frame, img_path.name)

        log_yolo.append(log_y)
        log_rtdetr.append(log_r)
        log_pipeline.append(log_p)

        pairs_yolo.append((img_path.name, res_y))
        pairs_rtdetr.append((img_path.name, res_r))
        pairs_pipeline.append((img_path.name, res_p))   # ← res_p = result สุดท้ายของ pipeline

    # DataFrame
    df_yolo     = pd.DataFrame(log_yolo)
    df_rtdetr   = pd.DataFrame(log_rtdetr)
    df_pipeline = pd.DataFrame(log_pipeline)

    # สรุปแต่ละแบบ
    summarize(df_yolo,     "YOLO อย่างเดียว")
    summarize(df_rtdetr,   "RT-DETR อย่างเดียว")
    summarize(df_pipeline, "Pipeline (YOLO + RT-DETR)")

    # บันทึก + เปรียบเทียบ
    save_csv(df_yolo, df_rtdetr, df_pipeline)
    compare_and_save(pairs_yolo, pairs_rtdetr, pairs_pipeline)


if __name__ == "__main__":
    main()