# ========== display.py ==========

import cv2
from config import COLOR_DAMAGE, COLOR_NORMAL, COLOR_INFO, WINDOW_NAME


def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["xyxy"])
        label = det["class_name"]
        conf  = det["conf"]
        color = COLOR_DAMAGE if "damage" in label else COLOR_NORMAL

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text   = f"{label} {conf:.2f}"
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        cv2.putText(frame, text, (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def draw_verdict(frame, verdict, fps, stage2_used, overlap_pct=0.0):
    h, w = frame.shape[:2]

    if verdict == "DAMAGE":
        color = COLOR_DAMAGE
        text  = "!! DAMAGE DETECTED !!"
    elif verdict == "NORMAL":
        color = COLOR_NORMAL
        text  = "NORMAL"
    else:
        color = COLOR_INFO
        text  = verdict if verdict else "BACKGROUND"
    
    cv2.rectangle(frame, (0, 0), (w, 50), color, -1)
    cv2.putText(frame, text, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    f_scale = 1.0
    f_thick = 3

    if stage2_used:
        # ใช้เครื่องหมาย | เพื่อแยกส่วนให้ดูเป็นระเบียบ
        full_text = f"Stage2: ON  |  Damage: {overlap_pct * 100:>5.1f}%"
        color = COLOR_INFO
    else:
        # ใส่ช่องว่าง (Spaces) เพิ่มเติมเพื่อให้ความยาวใกล้เคียงกับตอน ON
        full_text = f"Stage2: OFF |  Damage: {overlap_pct * 100:>5.1f}%"
        color = (200, 200, 200)

    # คำนวณกึ่งกลางจาก full_text
    t_size = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_thick)[0]
    t_x = (w - t_size[0]) // 2
    
    cv2.putText(frame, full_text, (t_x, h - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, f_scale, color, f_thick)

    return frame


def alert_terminal(verdict, detections, frame_count):
    if verdict == "DAMAGE":
        damage_dets = [d for d in detections if "damage" in d["class_name"]]
        print(f"\n{'!'*50}")
        print(f"  DAMAGE DETECTED — Frame #{frame_count}")
        print(f"  พบ damage : {len(damage_dets)} จุด")
        for i, d in enumerate(damage_dets, 1):
            print(f"    [{i}] conf={d['conf']:.2f}  box={[int(x) for x in d['xyxy']]}")
        print(f"{'!'*50}\n")


def show_frame(frame):
    cv2.imshow(WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF