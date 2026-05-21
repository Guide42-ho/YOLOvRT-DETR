import os
from roboflow import Roboflow

# --- ตั้งค่า Roboflow ---
# แนะนำให้ใช้ API Key จากหน้า Workspace ของคุณ
rf = Roboflow(api_key="IX2OWHwyaVaCqja3syDl") 
project = rf.workspace("guide-iogqi").project("label_bottle-syru7")

image_folder = 'extracted_frames/video-19v2' # โฟลเดอร์ที่เก็บภาพที่ต้องการอัปโหลด

# รายชื่อไฟล์ในโฟลเดอร์
files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"กำลังเริ่มอัปโหลดภาพจำนวน {len(files)} ภาพ...")

for index, filename in enumerate(files):
    file_path = os.path.join(image_folder, filename)
    
    try:
        # อัปโหลดทีละภาพ
        project.single_upload(
            image_path=file_path,
            batch_name="video-19v2" # ตั้งชื่อ Batch เพื่อให้หาใน Roboflow ง่าย
        )
        # print(f"[{index+1}/{len(files)}] Uploaded: {filename}")
    except Exception as e:
        print(f"Error uploading {filename}: {e}")

print("--- อัปโหลดครบถ้วนเรียบร้อย! ---")