import cv2
import os

# --- ตั้งค่าตรงนี้ ---
video_path = 'videos/video-26.MOV'  # ชื่อไฟล์วิดีโอ
output_folder = 'extracted_frames/video-26' # โฟลเดอร์ที่เก็บภาพ
frame_step = 15  # ดึงภาพทุกๆ X เฟรม (เช่น ทุกๆ 30 เฟรม)
# ------------------

# สร้างโฟลเดอร์ถ้ายังไม่มี
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# เปิดวิดีโอ
cap = cv2.VideoCapture(video_path)
current_frame = 0
saved_count = 0

print("กำลังเริ่มดึงภาพ...")

while True:
    # อ่านเฟรม
    success, frame = cap.read()
    
    # ถ้าหมดวิดีโอให้หยุด
    if not success:
        break
    
    # ตรวจสอบว่าครบรอบเฟรมที่ต้องการหรือยัง (ใช้การหารเอาเศษ)
    if current_frame % frame_step == 0:
        file_name = f"{output_folder}/video-26_frame_{current_frame}.jpg"
        cv2.imwrite(file_name, frame)
        saved_count += 1
        # print(f"บันทึกแล้ว: {file_name}")

    current_frame += 1

# ปิดการเชื่อมต่อ
cap.release()
print(f"--- เสร็จสิ้น! บันทึกภาพทั้งหมด {saved_count} ภาพ ---")