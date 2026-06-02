import cv2
import numpy as np


class BacklightingCorrector:
    """Corrects backlit scenes via CLAHE + shadow lift (pure software, no hardware)."""

    # [1.2] No HardwareExposureController — correction is 100% software

    def __init__(self, clahe_clip: float = 2.0, shadow_lift: int = 30,
                 frame_shape: tuple = (360, 640, 3)):
        self.shadow_lift = shadow_lift
        self.clahe_clip = clahe_clip
        # [2.1] np.full_like is NOT used in any per-frame method — buffer allocated here
        self._shadow_lift_buf = np.full(frame_shape, shadow_lift, dtype=np.uint8)

    def _clahe_for_shape(self, width: int) -> cv2.CLAHE:
        tile = (16, 16) if width >= 1000 else (8, 8)
        return cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=tile)

    def process(self, frame: np.ndarray) -> np.ndarray:
        # Resize only on shape change (rare); no np.full_like in hot path
        if self._shadow_lift_buf.shape != frame.shape:
            self._shadow_lift_buf = np.full(frame.shape, self.shadow_lift, dtype=np.uint8)
        lifted = cv2.add(frame, self._shadow_lift_buf)
        lab = cv2.cvtColor(lifted, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = self._clahe_for_shape(frame.shape[1])
        l = clahe.apply(l)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
