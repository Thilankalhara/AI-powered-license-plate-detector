"""
Configuration file for License Plate Detector
"""

# Model Configuration
YOLO_MODEL = "yolov8n.pt"  # YOLOv8 Nano (small, fast)
# Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
# Larger models = slower but more accurate

OCR_LANGUAGES = ['en']  # Supported languages for OCR

# Detection Parameters
CONFIDENCE_THRESHOLD = 0.5  # 0.0-1.0, lower = more detections
PLATE_SIMILARITY = 0.75  # Fuzzy matching threshold for plate comparison

# Video Processing
MAX_FRAME_WIDTH = 640
MAX_FRAME_HEIGHT = 480
VIDEO_FPS = 30

# File Paths
BLACKLIST_FILE = "blacklist.csv"
DETECTED_FILE = "detected_vehicles.csv"
LOG_FILE = "detection.log"

# GUI Settings
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900
THEME_COLOR_PRIMARY = "#4CAF50"
THEME_COLOR_ALERT = "#FF5252"
THEME_COLOR_BG = "#f5f5f5"

# Processing Options
USE_GPU = True  # Auto-detect CUDA if available
NUM_WORKERS = 4  # Number of threads for processing
BATCH_SIZE = 1  # Process N frames at once

# Appearance
SHOW_BOUNDING_BOXES = True
SHOW_CONFIDENCE_SCORES = True
HIGHLIGHT_DETECTED_PLATES = True
BOX_COLOR_NORMAL = (0, 255, 0)  # Green - BGR format
BOX_COLOR_DETECTED = (0, 0, 255)  # Red - BGR format
