"""
SafeVision AI - Feature Extraction Module
============================================
Uses MediaPipe Pose to detect human body landmarks, extracts
head and torso ROIs, and computes HSV color-space features
for PPE classification.
"""

import cv2
import numpy as np
import mediapipe as mp

from src import config
from src.utils import safe_crop, preprocess_roi


import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# ─────────────────────────────────────────────────
# MediaPipe Pose Landmark Indices
# (see https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
# ─────────────────────────────────────────────────
_NOSE = 0
_LEFT_EYE_INNER = 1
_LEFT_EYE = 2
_LEFT_EYE_OUTER = 3
_RIGHT_EYE_INNER = 4
_RIGHT_EYE = 5
_RIGHT_EYE_OUTER = 6
_LEFT_EAR = 7
_RIGHT_EAR = 8
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24


class PoseDetector:
    """
    Wrapper around MediaPipe Tasks PoseLandmarker that:
      1. Detects multiple people's pose landmarks.
      2. Derives head and torso bounding boxes from those landmarks.
      3. Extracts HSV color features from each ROI.
    """

    def __init__(self):
        self._ensure_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=config.POSE_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=5, # Detect up to 5 people
            min_pose_detection_confidence=config.POSE_MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.POSE_MIN_TRACKING_CONFIDENCE,
            output_segmentation_masks=False)
            
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def _ensure_model_exists(self):
        """Download the MediaPipe task model if it doesn't exist."""
        if not os.path.exists(config.POSE_MODEL_PATH):
            print(f"[PoseDetector] Downloading MediaPipe model to {config.POSE_MODEL_PATH}...")
            urllib.request.urlretrieve(config.POSE_MODEL_URL, config.POSE_MODEL_PATH)

    # ── Public API ──────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Detect people in a frame and extract PPE features.

        Returns a list of person dicts, each containing:
          - head_bbox, torso_bbox : (x1, y1, x2, y2) or None
          - head_features         : np.ndarray of shape (NUM_HEAD_FEATURES,)
          - torso_features        : np.ndarray of shape (NUM_TORSO_FEATURES,)
          - features              : concatenated feature vector
          - landmarks             : raw MediaPipe landmarks (for drawing)
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        detection_result = self._detector.detect(mp_image)
        persons = []
        
        if detection_result.pose_landmarks:
            for landmarks in detection_result.pose_landmarks:
                person = self._extract_person(frame, landmarks, w, h)
                if person is not None:
                    person["landmarks"] = landmarks
                    persons.append(person)

        return persons

    def draw_landmarks(self, frame: np.ndarray, landmarks) -> None:
        """Draw MediaPipe pose landmarks on the frame."""
        for lm in landmarks:
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            cv2.circle(frame, (x, y), 2, (0, 255, 200), -1)

    def release(self) -> None:
        """Release MediaPipe resources."""
        self._detector.close()

    # ── Internal Methods ────────────────────────────

    def _extract_person(self, frame: np.ndarray, landmarks: list,
                        img_w: int, img_h: int) -> dict | None:
        """Extract ROIs and features for a single detected person."""

        # ── Compute Head Bounding Box ───────────────
        head_bbox = self._compute_head_bbox(landmarks, img_w, img_h)
        # ── Compute Torso Bounding Box ──────────────
        torso_bbox = self._compute_torso_bbox(landmarks, img_w, img_h)

        if head_bbox is None and torso_bbox is None:
            return None

        # ── Extract ROIs & Features ─────────────────
        head_features = self._extract_head_features(frame, head_bbox)
        torso_features = self._extract_torso_features(frame, torso_bbox)

        # ── Concatenate into a single feature vector ─
        features = np.concatenate([head_features, torso_features])

        return {
            "head_bbox": head_bbox,
            "torso_bbox": torso_bbox,
            "head_features": head_features,
            "torso_features": torso_features,
            "features": features,
        }

    # ── Bounding Box Computation ────────────────────

    def _compute_head_bbox(self, lm, w: int, h: int) -> tuple | None:
        """
        Derive the head bounding box from facial landmark positions.
        Expands the box upward to capture a potential helmet.
        """
        indices = [_NOSE, _LEFT_EYE, _RIGHT_EYE,
                   _LEFT_EYE_INNER, _RIGHT_EYE_INNER,
                   _LEFT_EYE_OUTER, _RIGHT_EYE_OUTER,
                   _LEFT_EAR, _RIGHT_EAR]

        xs, ys = [], []
        for idx in indices:
            if lm[idx].visibility > 0.3:
                xs.append(lm[idx].x * w)
                ys.append(lm[idx].y * h)

        if len(xs) < 3:
            return None

        cx = np.mean(xs)
        cy = np.mean(ys)
        half_w = (max(xs) - min(xs)) / 2 * config.HEAD_ROI_EXPAND
        half_h = (max(ys) - min(ys)) / 2 * config.HEAD_ROI_EXPAND

        # Make the box taller upward (helmet extends above head)
        x1 = cx - half_w * 1.3
        y1 = cy - half_h * 2.5   # extend upward significantly
        x2 = cx + half_w * 1.3
        y2 = cy + half_h * 0.8

        return (x1, y1, x2, y2)

    def _compute_torso_bbox(self, lm, w: int, h: int) -> tuple | None:
        """
        Derive the torso bounding box from shoulder and hip landmarks.
        """
        indices = [_LEFT_SHOULDER, _RIGHT_SHOULDER, _LEFT_HIP, _RIGHT_HIP]
        xs, ys = [], []
        for idx in indices:
            if lm[idx].visibility > 0.3:
                xs.append(lm[idx].x * w)
                ys.append(lm[idx].y * h)

        if len(xs) < 3:
            return None

        padding_x = (max(xs) - min(xs)) * (config.TORSO_ROI_EXPAND - 1) / 2
        padding_y = (max(ys) - min(ys)) * (config.TORSO_ROI_EXPAND - 1) / 2

        x1 = min(xs) - padding_x
        y1 = min(ys) - padding_y
        x2 = max(xs) + padding_x
        y2 = max(ys) + padding_y

        return (x1, y1, x2, y2)

    # ── Feature Extraction ──────────────────────────

    def _extract_head_features(self, frame: np.ndarray,
                               bbox: tuple | None) -> np.ndarray:
        """
        Extract HSV color features from the head ROI.

        Features (10 total):
          [0-2] color_ratio for yellow, orange, white
          [3-5] mean H, S, V
          [6-8] std H, S, V
          [9]   dominant color ratio (max of the three color ratios)
        """
        if bbox is None:
            return np.zeros(config.NUM_HEAD_FEATURES, dtype=np.float32)

        roi = safe_crop(frame, bbox)
        if roi is None or roi.size == 0:
            return np.zeros(config.NUM_HEAD_FEATURES, dtype=np.float32)

        roi = preprocess_roi(roi)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        if total_pixels == 0:
            return np.zeros(config.NUM_HEAD_FEATURES, dtype=np.float32)

        # Color ratios
        ratios = []
        for name in ["yellow", "orange", "white"]:
            lo, hi = config.HELMET_HSV_RANGES[name]
            mask = cv2.inRange(hsv, lo, hi)
            ratios.append(np.count_nonzero(mask) / total_pixels)

        # HSV statistics
        mean_hsv = hsv.mean(axis=(0, 1))
        std_hsv = hsv.std(axis=(0, 1))

        # Dominant color ratio
        dominant = max(ratios)

        features = np.array(
            ratios + mean_hsv.tolist() + std_hsv.tolist() + [dominant],
            dtype=np.float32
        )
        return features

    def _extract_torso_features(self, frame: np.ndarray,
                                bbox: tuple | None) -> np.ndarray:
        """
        Extract HSV color features from the torso ROI.

        Features (11 total):
          [0-3] color_ratio for yellow, green, orange, fluorescent
          [4-6] mean H, S, V
          [7-9] std H, S, V
          [10]  dominant color ratio
        """
        if bbox is None:
            return np.zeros(config.NUM_TORSO_FEATURES, dtype=np.float32)

        roi = safe_crop(frame, bbox)
        if roi is None or roi.size == 0:
            return np.zeros(config.NUM_TORSO_FEATURES, dtype=np.float32)

        roi = preprocess_roi(roi)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        if total_pixels == 0:
            return np.zeros(config.NUM_TORSO_FEATURES, dtype=np.float32)

        # Color ratios
        ratios = []
        for name in ["yellow", "green", "orange", "fluorescent"]:
            lo, hi = config.VEST_HSV_RANGES[name]
            mask = cv2.inRange(hsv, lo, hi)
            ratios.append(np.count_nonzero(mask) / total_pixels)

        # HSV statistics
        mean_hsv = hsv.mean(axis=(0, 1))
        std_hsv = hsv.std(axis=(0, 1))

        # Dominant color ratio
        dominant = max(ratios)

        features = np.array(
            ratios + mean_hsv.tolist() + std_hsv.tolist() + [dominant],
            dtype=np.float32
        )
        return features
