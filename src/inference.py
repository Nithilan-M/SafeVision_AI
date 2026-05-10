"""
SafeVision AI - Real-Time Inference Engine
=============================================
Runs the trained PPE classifier on webcam, video file,
or static image input with full annotation overlays.
"""

import os
import sys
import argparse

import cv2
import numpy as np
import joblib

from src import config
from src.feature_extraction import PoseDetector
from src.utils import (
    FPSCounter, draw_detection_results,
    draw_warning_banner, draw_fps
)
from src.logger import DetectionLogger


class PPEInferenceEngine:
    """Loads the trained model and runs real-time PPE detection."""

    def __init__(self, model_path=None, scaler_path=None):
        model_path = model_path or config.MODEL_PATH
        scaler_path = scaler_path or config.SCALER_PATH

        if not os.path.exists(model_path):
            print(f"[Inference] ERROR: Model not found at {model_path}")
            sys.exit(1)

        payload = joblib.load(model_path)
        self._helmet_model = payload["helmet_model"]
        self._vest_model = payload["vest_model"]
        self._scaler = joblib.load(scaler_path)
        self._detector = PoseDetector()
        self._fps = FPSCounter()
        self._logger = DetectionLogger()
        self._frame_count = 0
        print("[Inference] Engine initialized.")

    def run_on_image(self, image_path, output_path=None, show=True):
        """Run PPE detection on a single image."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"[Inference] ERROR: Cannot read {image_path}")
            return None
        annotated = self._process_frame(frame)
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, annotated)
            print(f"[Inference] Saved -> {output_path}")
        if show:
            cv2.imshow("SafeVision AI", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return annotated

    def run_on_video(self, source=0, output_path=None, show=True):
        """Run PPE detection on video stream or webcam."""
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Inference] ERROR: Cannot open {source}")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_n = cap.get(cv2.CAP_PROP_FPS) or 30.0

        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            writer = cv2.VideoWriter(output_path,
                cv2.VideoWriter_fourcc(*"mp4v"), fps_n, (w, h))

        self._logger.open()
        print(f"[Inference] Running on {'Webcam' if source == 0 else source}")
        print("[Inference] Press 'q' to quit.\n")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = self._process_frame(frame)
                if writer:
                    writer.write(annotated)
                if show:
                    cv2.imshow("SafeVision AI", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            self._logger.close()
            self._detector.release()
        print(f"[Inference] Processed {self._frame_count} frames.")

    def _process_frame(self, frame):
        """Run detection, classification, and annotation."""
        self._frame_count += 1
        annotated = frame.copy()
        persons = self._detector.process_frame(frame)
        violation = False

        for pid, person in enumerate(persons):
            features = person["features"].reshape(1, -1)
            features_scaled = self._scaler.transform(features)

            helmet_pred = self._helmet_model.predict(features_scaled)[0]
            vest_pred = self._vest_model.predict(features_scaled)[0]

            h_conf = (self._helmet_model.predict_proba(features_scaled)[0][1]
                      if hasattr(self._helmet_model, "predict_proba") else 0.0)
            v_conf = (self._vest_model.predict_proba(features_scaled)[0][1]
                      if hasattr(self._vest_model, "predict_proba") else 0.0)

            result = {
                "head_bbox": person["head_bbox"],
                "torso_bbox": person["torso_bbox"],
                "helmet": bool(helmet_pred),
                "vest": bool(vest_pred),
                "helmet_conf": h_conf, "vest_conf": v_conf,
            }
            draw_detection_results(annotated, result)
            if "landmarks" in person:
                self._detector.draw_landmarks(annotated, person["landmarks"])
            if config.LOG_DETECTIONS:
                self._logger.log(self._frame_count, pid, result)
            if not helmet_pred or not vest_pred:
                violation = True

        if violation:
            draw_warning_banner(annotated, "PPE VIOLATION — Safety equipment missing!")
        draw_fps(annotated, self._fps.tick())
        return annotated


def main():
    parser = argparse.ArgumentParser(description="SafeVision AI Inference")
    parser.add_argument("--source", default="0", help="0=webcam or file path")
    parser.add_argument("--output", default=None, help="Save annotated output")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--scaler", default=None)
    args = parser.parse_args()

    engine = PPEInferenceEngine(args.model, args.scaler)
    source = args.source
    show = not args.no_display
    try:
        source = int(source)
    except ValueError:
        pass

    if isinstance(source, str) and source.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
        out = args.output or os.path.join(config.OUTPUT_DIR,
            "detected_" + os.path.basename(source))
        engine.run_on_image(source, output_path=out, show=show)
    else:
        out = args.output
        if out is None and isinstance(source, str):
            out = os.path.join(config.OUTPUT_DIR,
                "detected_" + os.path.basename(source))
        engine.run_on_video(source, output_path=out, show=show)


if __name__ == "__main__":
    main()
