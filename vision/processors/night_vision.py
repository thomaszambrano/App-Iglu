import cv2
import numpy as np


class NightVisionProcessor:
    """Applies gamma correction + CLAHE exclusively to the L channel of LAB."""

    def __init__(self, gamma: float = 1.8, clahe_clip: float = 2.0):
        self.gamma = gamma
        self.clahe_clip = clahe_clip
        self._lut = self._build_lut(gamma)

    def _build_lut(self, gamma: float) -> np.ndarray:
        inv = 1.0 / gamma
        table = (np.arange(256) / 255.0) ** inv * 255.0
        return np.clip(table, 0, 255).astype(np.uint8)

    def reconfigure(self, gamma: float) -> None:
        """Rebuild the LUT in-place. Changing gamma without calling this has no effect."""
        self.gamma = gamma
        self._lut = self._build_lut(gamma)

    def _clahe_for_shape(self, width: int) -> cv2.CLAHE:
        tile = (16, 16) if width >= 1000 else (8, 8)
        return cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=tile)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """BGR → LAB, gamma+CLAHE on L only → BGR. A/B channels are untouched."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # [1.1] gamma LUT applied only to L channel
        l = cv2.LUT(l, self._lut)
        clahe = self._clahe_for_shape(frame.shape[1])
        l = clahe.apply(l)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
