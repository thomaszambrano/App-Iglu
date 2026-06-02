"""Live Laplacian variance display + threshold recommendation after 30 seconds."""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import cv2
import numpy as np

from vision.utils.blur import BlurCheck


def main():
    checker = BlurCheck()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera 0")
        sys.exit(1)

    variances = []
    start = time.time()
    duration = 30.0
    print("Streaming Laplacian variance for 30 seconds. Press Ctrl+C to stop early.")

    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration:
                break
            ok, frame = cap.read()
            if not ok:
                break
            v = checker.laplacian_variance(frame)
            variances.append(v)
            print(f"[{elapsed:5.1f}s] variance={v:.2f}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    if variances:
        p10 = float(np.percentile(variances, 10))
        print(f"\n--- Calibration Result ---")
        print(f"Frames captured : {len(variances)}")
        print(f"10th percentile : {p10:.2f}")
        print(f"Recommendation  : set blur_threshold = {p10:.1f} in your config")
    else:
        print("No frames captured.")


if __name__ == "__main__":
    main()
