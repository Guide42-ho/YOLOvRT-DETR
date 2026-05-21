# ========== main.py ==========

import cv2
import time
from models   import load_models
from detector import detect_stage1, detect_stage2
from decision import has_damage, pipeline_decision
from display  import draw_boxes, draw_verdict, alert_terminal
from config   import CAM_INDEX, OUTPUT_VIDEO_PATH


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("❌ เปิดกล้องไม่ได้ เช็ค CAM_INDEX ใน config.py")
    return cap


def main():
    print("=" * 50)
    print("  Bottle Label Inspection — Video Recording")
    print(f"  บันทึกผลที่ : {OUTPUT_VIDEO_PATH}")
    print("=" * 50)

    model_yolo, model_rtdetr = load_models()
    print("\n✅ พร้อมทำงาน — เปิดกล้องแล้ว\n")

    cap     = open_camera()
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps_src, (w, h))

    frame_count = 0
    t_prev      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ อ่านภาพจากกล้องไม่ได้")
            break

        frame_count += 1

        stage1_dets = detect_stage1(model_yolo, frame)
        stage2_used = False
        stage2_dets = []

        if has_damage(stage1_dets):
            stage2_used = True
            stage2_dets = detect_stage2(model_rtdetr, frame)
        
        stage_final = stage2_dets if stage2_used else stage1_dets
        

        verdict, final_dets, overlap_pct = pipeline_decision(stage_final, frame.shape)


        t_now  = time.time()
        fps    = 1.0 / (t_now - t_prev + 1e-6)
        t_prev = t_now

        frame = draw_boxes(frame, final_dets)
        frame = draw_verdict(frame, verdict, fps, stage2_used, overlap_pct)

        # if verdict == "DAMAGE":
        #     alert_terminal(verdict, final_dets, frame_count)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\n✅ บันทึกวิดีโอเรียบร้อย → {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()