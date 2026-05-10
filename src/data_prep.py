"""
SafeVision AI - Data Preparation Module
==========================================
Handles two workflows:
  1. Process a labeled image dataset (Hard Hat Workers / SHWD) to extract
     features and build a training CSV.
  2. Generate synthetic training data from color-augmented dummy images
     so the pipeline can be tested end-to-end without a large dataset.
"""

import os
import csv
import glob
import random

import cv2
import numpy as np
import pandas as pd

from src import config
from src.feature_extraction import PoseDetector


# ─────────────────────────────────────────────────
# Synthetic Data Generator
# ─────────────────────────────────────────────────

def generate_synthetic_features(n_samples: int = 2000,
                                 save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic feature vectors that mimic what the real
    pipeline would extract, with known ground-truth labels.

    This lets us train and test the ML model before a full dataset
    is available.

    Returns:
        DataFrame with columns = feature_0 .. feature_N, helmet, vest
    """
    save_path = save_path or config.FEATURES_CSV
    rng = np.random.default_rng(42)

    rows = []
    for _ in range(n_samples):
        helmet = rng.choice([0, 1])
        vest = rng.choice([0, 1])

        # ── Head features (10) ──────────────────────
        if helmet:
            # PPE present -> higher yellow/orange ratios, brighter
            color_yellow  = rng.uniform(0.15, 0.55)
            color_orange  = rng.uniform(0.05, 0.35)
            color_white   = rng.uniform(0.0, 0.15)
            mean_h = rng.uniform(18, 35)
            mean_s = rng.uniform(120, 220)
            mean_v = rng.uniform(150, 240)
        else:
            # No helmet -> low safety-color ratios
            color_yellow  = rng.uniform(0.0, 0.08)
            color_orange  = rng.uniform(0.0, 0.06)
            color_white   = rng.uniform(0.0, 0.10)
            mean_h = rng.uniform(0, 180)
            mean_s = rng.uniform(20, 120)
            mean_v = rng.uniform(40, 180)

        std_h = rng.uniform(5, 40)
        std_s = rng.uniform(15, 60)
        std_v = rng.uniform(15, 60)
        dominant_head = max(color_yellow, color_orange, color_white)

        head_feats = [color_yellow, color_orange, color_white,
                      mean_h, mean_s, mean_v,
                      std_h, std_s, std_v, dominant_head]

        # ── Torso features (11) ─────────────────────
        if vest:
            t_yellow      = rng.uniform(0.10, 0.50)
            t_green       = rng.uniform(0.05, 0.35)
            t_orange      = rng.uniform(0.05, 0.30)
            t_fluorescent = rng.uniform(0.05, 0.40)
            t_mean_h = rng.uniform(25, 50)
            t_mean_s = rng.uniform(130, 230)
            t_mean_v = rng.uniform(140, 240)
        else:
            t_yellow      = rng.uniform(0.0, 0.07)
            t_green       = rng.uniform(0.0, 0.06)
            t_orange      = rng.uniform(0.0, 0.05)
            t_fluorescent = rng.uniform(0.0, 0.05)
            t_mean_h = rng.uniform(0, 180)
            t_mean_s = rng.uniform(10, 110)
            t_mean_v = rng.uniform(30, 170)

        t_std_h = rng.uniform(5, 40)
        t_std_s = rng.uniform(15, 60)
        t_std_v = rng.uniform(15, 60)
        dominant_torso = max(t_yellow, t_green, t_orange, t_fluorescent)

        torso_feats = [t_yellow, t_green, t_orange, t_fluorescent,
                       t_mean_h, t_mean_s, t_mean_v,
                       t_std_h, t_std_s, t_std_v, dominant_torso]

        row = head_feats + torso_feats + [helmet, vest]
        rows.append(row)

    # Build DataFrame
    feature_cols = [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]
    columns = feature_cols + ["helmet", "vest"]
    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[DataPrep] Saved {len(df)} synthetic samples -> {save_path}")
    return df


# ─────────────────────────────────────────────────
# Real Dataset Processor
# ─────────────────────────────────────────────────

def process_image_dataset(image_dir: str, label_dir: str = None,
                          label_format: str = "yolo",
                          save_path: str = None) -> pd.DataFrame:
    """
    Process a labeled image dataset to extract features.

    For simplicity, this supports a folder of images with
    corresponding YOLO-format .txt label files, where class 0 = helmet,
    class 1 = no-helmet, class 2 = vest, class 3 = no-vest.

    Args:
        image_dir:    Path to folder containing images.
        label_dir:    Path to folder containing label .txt files.
                      Defaults to same as image_dir.
        label_format: "yolo" (only format supported currently).
        save_path:    Where to save the extracted features CSV.

    Returns:
        DataFrame of extracted features with labels.
    """
    save_path = save_path or config.FEATURES_CSV
    label_dir = label_dir or image_dir

    detector = PoseDetector()
    rows = []

    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg")) +
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpeg"))
    )

    print(f"[DataPrep] Processing {len(image_paths)} images from {image_dir}")

    for idx, img_path in enumerate(image_paths):
        if idx % 100 == 0:
            print(f"  ... processed {idx}/{len(image_paths)}")

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # ── Parse labels ────────────────────────────
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        helmet_label, vest_label = _parse_yolo_labels(label_path)

        # ── Extract features via MediaPipe ──────────
        persons = detector.process_frame(frame)
        if not persons:
            continue

        # Use the first detected person
        person = persons[0]
        features = person["features"]

        row = features.tolist() + [helmet_label, vest_label]
        rows.append(row)

    detector.release()

    # Build DataFrame
    feature_cols = [f"feature_{i}" for i in range(config.TOTAL_FEATURES)]
    columns = feature_cols + ["helmet", "vest"]
    df = pd.DataFrame(rows, columns=columns)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[DataPrep] Saved {len(df)} real samples -> {save_path}")
    return df


def _parse_yolo_labels(label_path: str) -> tuple[int, int]:
    """
    Parse YOLO-format labels to determine helmet and vest presence.

    Expected class IDs:
        0 = helmet present
        1 = head (no helmet)
        2 = vest present
        3 = person (no vest)

    Returns:
        (helmet: 0 or 1, vest: 0 or 1)
    """
    helmet = 0
    vest = 0

    if not os.path.exists(label_path):
        return helmet, vest

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id == 0:
                helmet = 1
            elif class_id == 2:
                vest = 1

    return helmet, vest


# ─────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SafeVision AI - Data Preparation")
    parser.add_argument("--mode", choices=["synthetic", "dataset"],
                        default="synthetic",
                        help="'synthetic' to generate dummy data, "
                             "'dataset' to process real images.")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Path to image directory (for dataset mode).")
    parser.add_argument("--label-dir", type=str, default=None,
                        help="Path to label directory (defaults to image-dir).")
    parser.add_argument("--n-samples", type=int, default=2000,
                        help="Number of synthetic samples to generate.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path.")

    args = parser.parse_args()

    if args.mode == "synthetic":
        generate_synthetic_features(n_samples=args.n_samples,
                                     save_path=args.output)
    else:
        if not args.image_dir:
            print("ERROR: --image-dir is required for dataset mode.")
        else:
            process_image_dataset(args.image_dir, args.label_dir,
                                  save_path=args.output)
