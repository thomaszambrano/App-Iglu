"""Synchronous dev loop for drowsiness detection with optional simulation overlay."""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2

from vision.config import LOCAL_TEST_CONFIG
from vision.pipeline import PreprocessingPipeline
from vision.detection.landmarks import FaceMeshDetector
from vision.detection.metrics import compute_ear, compute_mar
from vision.detection.alert import AlertManager
from vision.ambient import AmbientMode
from vision.utils.simulate import simulate_night_environment, simulate_backlit_environment


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", choices=["none", "night", "backlit"], default="none")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline = PreprocessingPipeline(LOCAL_TEST_CONFIG)
    detector = FaceMeshDetector()
    alert_mgr = AlertManager()

    # [run_dev] No FOURCC hint; default backend only
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("ERROR: Cannot open camera 0")
        sys.exit(1)

    _simulate_fns = {
        "night": simulate_night_environment,
        "backlit": simulate_backlit_environment,
    }

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.simulate in _simulate_fns:
            frame = _simulate_fns[args.simulate](frame)

        result = pipeline.process(frame)
        processed = result["frame"]
        mode: AmbientMode = result["mode"]

        ear, mar = 0.0, 0.0
        landmarks = detector.process(processed)
        if landmarks and not result["skip_inference"]:
            ear = compute_ear(landmarks)
            mar = compute_mar(landmarks)
            alert_mgr.update(ear, mar, mode)

        # Overlay
        overlay = processed.copy()
        y = 20
        lines = [
            f"Mode: {mode.name}",
            f"EAR: {ear:.3f}  MAR: {mar:.3f}",
            f"skip_inference: {result['skip_inference']}",
            f"ambient_ms: {result['ambient_ms']:.1f}",
            f"preprocess_ms: {result['preprocess_ms']:.1f}",
        ]
        for line in lines:
            cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
            y += 22

        cv2.imshow("Iglu Dev", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
