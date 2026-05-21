# ========== config.py ==========

# Path โมเดล
MODEL_YOLO_PATH   = "train_yolov8s/weights/best.pt"
MODEL_RTDETR_PATH = "train_rtdetr-l/weights/best.pt"

# ชื่อ class
DAMAGE_CLASS = "label_damage"
NORMAL_CLASS = "label_normal"

# Confidence threshold
CONF_STAGE1 = 0.45
CONF_STAGE2 = 0.64  

# Device
DEVICE = 0

# Webcam
CAM_INDEX  = "0519(1).mp4"      # กล้องตัวที่ 0
CAM_WIDTH  = 1280
CAM_HEIGHT = 720

# สี bounding box (BGR)
COLOR_DAMAGE = (0,   0,   255)  # แดง
COLOR_NORMAL = (0,   255,   0)  # เขียว
COLOR_INFO   = (255, 255,   0)  # เหลือง

# หน้าต่าง
WINDOW_NAME = "Bottle Label Inspection"