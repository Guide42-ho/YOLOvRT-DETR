# ========== config.py ==========

# Path โมเดล
MODEL_YOLO_PATH = "train_yolov8s/weights/best.pt"
MODEL_RTDETR_PATH = "train_rtdetr-l/weights/best.pt"

# Path dataset
TEST_DIR  = "Label_bottle-2/test/images"
DATA_YAML = "Label_bottle-2/data.yaml"
LABEL_DIR = "Label_bottle-2/test/labels"


# ชื่อ class
DAMAGE_CLASS = "label_damage"
NORMAL_CLASS = "label_normal"

# Confidence threshold

CONF_SOLO   = 0.3   # รันแยก (ไม่ใช้ pipeline)

# Device
DEVICE = 0  # GPU

# Output
OUTPUT_CSV = "pipeline_results.csv"

OUTPUT_CSV_YOLO = "results_yolo.csv"
OUTPUT_CSV_RTDETR = "results_rtdetr.csv"
OUTPUT_CSV_PIPELINE = "results_pipeline.csv"
OUTPUT_CSV_COMPARE = "results_compare.csv"