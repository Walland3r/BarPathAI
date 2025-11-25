"""
Configuration settings for BarPath AI app
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
PROJECT_DIR = BASE_DIR.parent
STORAGE_DIR = BASE_DIR / "storage"
TEMP_DIR = STORAGE_DIR / "temp"
DATA_DIR = STORAGE_DIR / "data"

# Ensure directories exist
TEMP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model settings
DEFAULT_MODEL_PATH = PROJECT_DIR / "best.pt"
MODEL_PATH = str(DEFAULT_MODEL_PATH)

# Video processing settings
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_IOU_THRESHOLD = 0.45
MIN_CYCLE_LENGTH = 30  # Minimum frames for a valid repetition
TRAIL_FADE_DURATION = 60  # Frames
BBOX_SMOOTHING_WINDOW = 5

# Video settings
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv"]
OUTPUT_VIDEO_CODEC = "mp4v"
SUMMARY_FRAME_DURATION = 10  # seconds

# UI settings
APP_TITLE = "BarPath AI - Barbell Tracker"
APP_THEME = "dark"
PROGRESS_UPDATE_INTERVAL = 10  # Update progress every N frames

# Colors (BGR format for OpenCV)
COLOR_TRAIL_DOWN = (0, 255, 0)  # Green
COLOR_TRAIL_UP = (0, 0, 255)    # Red
COLOR_CENTER_POINT = (255, 0, 0)  # Blue
COLOR_TEXT = (255, 255, 255)  # White