import cv2
import numpy as np


class BlurCheck:
    """Detects blurry frames using Laplacian variance on a downsampled image."""

    def __init__(self, threshold: float = 80.0):
        self.threshold = threshold

    def is_blurry(self, frame: np.ndarray) -> bool:
        # [2.2] Blur check runs on a 320×180 downsample, not full-res
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.threshold

    def laplacian_variance(self, frame: np.ndarray) -> float:
        small = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
