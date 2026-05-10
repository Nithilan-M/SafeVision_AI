"""
SafeVision AI - Utility Functions
===================================
Helper functions for image processing, drawing overlays,
and computing performance metrics.
"""

import cv2
import numpy as np
import time
from src import config


class FPSCounter:
    """Tracks and smooths FPS across frames using a rolling average."""

    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._timestamps: list[float] = []

    def tick(self) -> float:
        """Call once per frame. Returns the current smoothed FPS."""
        now = time.perf_counter()
        self._timestamps.append(now)
        # Keep only the last `window_size` timestamps
        if len(self._timestamps) > self._window_size:
            self._timestamps = self._timestamps[-self._window_size:]
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


def draw_label(frame: np.ndarray, text: str, position: tuple,
               color: tuple = config.COLOR_SAFE,
               bg_color: tuple = config.COLOR_TEXT_BG) -> None:
    """Draw a text label with a filled background rectangle."""
    x, y = int(position[0]), int(position[1])
    (tw, th), baseline = cv2.getTextSize(
        text, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS
    )
    # Background rectangle
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + baseline + 4),
                  bg_color, -1)
    # Text
    cv2.putText(frame, text, (x + 4, y - 2), config.FONT,
                config.FONT_SCALE, color, config.FONT_THICKNESS)


def draw_bbox(frame: np.ndarray, bbox: tuple, color: tuple,
              thickness: int = config.BOX_THICKNESS) -> None:
    """Draw a bounding box rectangle on the frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_warning_banner(frame: np.ndarray, message: str) -> None:
    """Draw a full-width warning banner at the top of the frame."""
    h, w = frame.shape[:2]
    banner_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, f"⚠  {message}", (15, 33), config.FONT,
                0.8, config.COLOR_WHITE, 2)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw the FPS counter in the top-right corner."""
    h, w = frame.shape[:2]
    text = f"FPS: {fps:.1f}"
    (tw, th), _ = cv2.getTextSize(text, config.FONT, 0.7, 2)
    x = w - tw - 15
    y = 30
    cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5),
                  (40, 40, 40), -1)
    cv2.putText(frame, text, (x, y), config.FONT, 0.7,
                (0, 255, 200), 2)


def draw_detection_results(frame: np.ndarray, person: dict) -> None:
    """
    Draw all detection annotations for one person.

    Args:
        frame:  The video/image frame to annotate.
        person: Dict with keys:
            - head_bbox: (x1, y1, x2, y2)
            - torso_bbox: (x1, y1, x2, y2)
            - helmet: bool
            - vest: bool
            - helmet_conf: float
            - vest_conf: float
    """
    helmet = person.get("helmet", False)
    vest = person.get("vest", False)
    h_conf = person.get("helmet_conf", 0.0)
    v_conf = person.get("vest_conf", 0.0)

    # --- Head bounding box + label ---
    head_bbox = person.get("head_bbox")
    if head_bbox is not None:
        h_color = config.COLOR_SAFE if helmet else config.COLOR_VIOLATION
        draw_bbox(frame, head_bbox, h_color)
        label = f"Helmet: {'YES' if helmet else 'NO'} ({h_conf:.0%})"
        draw_label(frame, label, (head_bbox[0], head_bbox[1] - 5), h_color)

    # --- Torso bounding box + label ---
    torso_bbox = person.get("torso_bbox")
    if torso_bbox is not None:
        v_color = config.COLOR_SAFE if vest else config.COLOR_VIOLATION
        draw_bbox(frame, torso_bbox, v_color)
        label = f"Vest: {'YES' if vest else 'NO'} ({v_conf:.0%})"
        draw_label(frame, label, (torso_bbox[0], torso_bbox[1] - 5), v_color)


def safe_crop(image: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """
    Safely crop a region from an image, clamping to image boundaries.
    Returns None if the resulting crop has zero area.
    """
    h, w = image.shape[:2]
    x1 = max(0, int(bbox[0]))
    y1 = max(0, int(bbox[1]))
    x2 = min(w, int(bbox[2]))
    y2 = min(h, int(bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing to an ROI to improve robustness under
    varying lighting conditions.
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Gaussian blur to reduce noise
    """
    # Convert to LAB for CLAHE on the L-channel
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Light Gaussian blur
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return enhanced
