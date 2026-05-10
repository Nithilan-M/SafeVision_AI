"""
SafeVision AI - Configuration Module
=====================================
Central configuration for HSV color ranges, model paths,
detection thresholds, and display settings.
"""

import os
import numpy as np

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Create directories if they don't exist
for d in [MODEL_DIR, LOG_DIR, DATA_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "ppe_classifier.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "feature_scaler.pkl")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")

# ─────────────────────────────────────────────
# MediaPipe Pose Configuration
# ─────────────────────────────────────────────
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
POSE_MIN_DETECTION_CONFIDENCE = 0.5
POSE_MIN_TRACKING_CONFIDENCE = 0.5

# ─────────────────────────────────────────────
# HSV Color Ranges for PPE Detection
# Each entry is (lower_bound, upper_bound) in HSV
# ─────────────────────────────────────────────

# --- Helmet Colors (Yellow / Orange / White) ---
HELMET_HSV_RANGES = {
    "yellow": (np.array([20, 100, 100]), np.array([35, 255, 255])),
    "orange": (np.array([10, 100, 100]), np.array([20, 255, 255])),
    "white":  (np.array([0, 0, 180]),    np.array([180, 50, 255])),
}

# --- Safety Vest Colors (Yellow / Green / Orange / Fluorescent) ---
VEST_HSV_RANGES = {
    "yellow":      (np.array([20, 100, 100]),  np.array([35, 255, 255])),
    "green":       (np.array([35, 80, 80]),    np.array([85, 255, 255])),
    "orange":      (np.array([10, 100, 100]),  np.array([20, 255, 255])),
    "fluorescent": (np.array([25, 150, 150]),  np.array([45, 255, 255])),
}

# ─────────────────────────────────────────────
# Detection Thresholds
# ─────────────────────────────────────────────
# Minimum percentage of PPE-colored pixels in the ROI to
# consider the equipment "present" (used as a heuristic fallback)
HELMET_COLOR_THRESHOLD = 0.10   # 10% of head ROI
VEST_COLOR_THRESHOLD = 0.15     # 15% of torso ROI

# ROI expansion factor (multiplied with landmark-based bounding box)
HEAD_ROI_EXPAND = 1.6   # Expand head box to capture helmet brim
TORSO_ROI_EXPAND = 1.2  # Slight expansion for vest edges

# ─────────────────────────────────────────────
# Feature Vector Configuration
# ─────────────────────────────────────────────
# Features extracted per ROI for the ML classifier
# Head features: [color_ratio_yellow, color_ratio_orange, color_ratio_white,
#                 mean_h, mean_s, mean_v, std_h, std_s, std_v, dominant_color_ratio]
# Torso features: [color_ratio_yellow, color_ratio_green, color_ratio_orange,
#                  color_ratio_fluorescent, mean_h, mean_s, mean_v,
#                  std_h, std_s, std_v, dominant_color_ratio]
NUM_HEAD_FEATURES = 10
NUM_TORSO_FEATURES = 11
TOTAL_FEATURES = NUM_HEAD_FEATURES + NUM_TORSO_FEATURES

# ─────────────────────────────────────────────
# Display / Visualization Settings
# ─────────────────────────────────────────────
# Colors are in BGR for OpenCV
COLOR_SAFE = (0, 200, 0)       # Green  – compliant
COLOR_VIOLATION = (0, 0, 220)  # Red    – non-compliant
COLOR_WARNING = (0, 180, 255)  # Orange – partial compliance
COLOR_TEXT_BG = (30, 30, 30)   # Dark background for text
COLOR_WHITE = (255, 255, 255)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
BOX_THICKNESS = 2

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
LOG_DETECTIONS = True
LOG_FILE = os.path.join(LOG_DIR, "detection_log.csv")
