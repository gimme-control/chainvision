# Configuration file for combined.py

# YOLO model paths
PERSON_MODEL_PATH = "models/yolov8n.pt"
GUN_MODEL_PATH = "models/best3.pt"

# Video capture settings
VIDEO_SOURCE = "luigi.mp4"
FRAME_WIDTH = 600
FRAME_HEIGHT = 300

# Detection thresholds
PERSON_CONF_THRESHOLD = 0.3
GUN_CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.01

# Re-identification settings
REID_SIMILARITY_THRESHOLD = 0.2  # Increased for robustness

# Server configuration
SERVER_URL = "http://localhost:8000"

# Embedding management
UPLOAD_INTERVAL = 30  # Number of frames between uploads
CACHE_FORCE_REFRESH = False

# Logging settings
LOGGING_LEVEL = "DEBUG"