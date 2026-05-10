"""
SafeVision AI - Detection Logger
==================================
Logs PPE detection results to CSV and console for auditing.
"""

import os
import csv
import json
from datetime import datetime
from src import config


class DetectionLogger:
    """Writes per-frame detection results to a CSV log file."""

    FIELDNAMES = [
        "timestamp", "frame_number", "person_id",
        "helmet_detected", "helmet_confidence",
        "vest_detected", "vest_confidence",
        "head_bbox", "torso_bbox",
        "status"
    ]

    def __init__(self, log_path: str = None):
        self._log_path = log_path or config.LOG_FILE
        self._file = None
        self._writer = None
        self._initialized = False

    # ── Context manager support ─────────────────────
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ── Core methods ─────────────────────────────────
    def open(self) -> None:
        """Open the CSV file and write headers if file is new."""
        file_exists = os.path.exists(self._log_path)
        self._file = open(self._log_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        if not file_exists:
            self._writer.writeheader()
        self._initialized = True

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file:
            self._file.close()
            self._file = None
        self._initialized = False

    def log(self, frame_number: int, person_id: int, person: dict) -> None:
        """
        Log a single person detection result.

        Args:
            frame_number: Current video frame index.
            person_id:    Index of the detected person in the frame.
            person:       Dict with detection results (helmet, vest, bboxes, confs).
        """
        if not self._initialized:
            self.open()

        helmet = person.get("helmet", False)
        vest = person.get("vest", False)

        # Determine compliance status
        if helmet and vest:
            status = "COMPLIANT"
        elif helmet or vest:
            status = "PARTIAL_VIOLATION"
        else:
            status = "VIOLATION"

        row = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": frame_number,
            "person_id": person_id,
            "helmet_detected": helmet,
            "helmet_confidence": round(person.get("helmet_conf", 0.0), 4),
            "vest_detected": vest,
            "vest_confidence": round(person.get("vest_conf", 0.0), 4),
            "head_bbox": json.dumps(person.get("head_bbox")),
            "torso_bbox": json.dumps(person.get("torso_bbox")),
            "status": status,
        }
        self._writer.writerow(row)
        self._file.flush()

        # Console output
        print(
            f"[{row['timestamp']}] Frame {frame_number:>5d} | "
            f"Person {person_id} | "
            f"Helmet: {'YES' if helmet else ' NO'} ({row['helmet_confidence']:.2f}) | "
            f"Vest:   {'YES' if vest else ' NO'} ({row['vest_confidence']:.2f}) | "
            f"Status: {status}"
        )
