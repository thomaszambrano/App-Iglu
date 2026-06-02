import cv2
import numpy as np
from enum import Enum, auto
from typing import Optional, Tuple


class AmbientMode(Enum):
    NORMAL = auto()
    NIGHT_VISION = auto()
    BACKLIT = auto()


class AmbientDetector:
    """Samples ambient light conditions and maintains hysteresis-filtered mode."""

    def __init__(self, config: dict):
        self._night_trigger = config["night_vision_trigger_luma"]
        self._backlit_ratio = config["backlit_trigger_ratio"]
        self._hysteresis = config["hysteresis_buffer"]
        self._mode = AmbientMode.NORMAL
        self._votes: dict[AmbientMode, int] = {m: 0 for m in AmbientMode}
        # [3.3] Guarantee first-frame check unconditionally
        self._force_next = True

    def should_sample(self, frame_idx: int) -> bool:
        if self._force_next:
            self._force_next = False
            return True
        return frame_idx % 15 == 0

    def _classify_frame(
        self, frame: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None
    ) -> AmbientMode:
        # [3.1] Convert a 0.25× downsampled frame to LAB — never full-res
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (max(1, w // 4), max(1, h // 4)))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        global_luma = float(np.mean(l_channel))

        if global_luma < self._night_trigger:
            return AmbientMode.NIGHT_VISION

        # [3.2] Use face_box for center crop when available; fall back to center 35%
        # Fallback assumption: face is roughly centered in frame.
        sh, sw = small.shape[:2]
        if face_box is not None:
            fx, fy, fw, fh = face_box
            # Scale box coords to the downsampled frame
            sx = int(fx * sw / w)
            sy = int(fy * sh / h)
            sfw = max(1, int(fw * sw / w))
            sfh = max(1, int(fh * sh / h))
            center_region = l_channel[sy:sy + sfh, sx:sx + sfw]
        else:
            cy, cx = sh // 2, sw // 2
            rh, rw = int(sh * 0.35 / 2), int(sw * 0.35 / 2)
            center_region = l_channel[cy - rh:cy + rh, cx - rw:cx + rw]

        center_luma = float(np.mean(center_region)) if center_region.size > 0 else global_luma

        # [1.3] Guard against near-zero center luma
        if center_luma < 15:
            return AmbientMode.NORMAL

        bg_luma = global_luma
        ratio = bg_luma / center_luma if center_luma > 0 else 1.0

        if ratio >= self._backlit_ratio:
            return AmbientMode.BACKLIT

        return AmbientMode.NORMAL

    def update(
        self, frame: np.ndarray, frame_idx: int,
        face_box: Optional[Tuple[int, int, int, int]] = None
    ) -> AmbientMode:
        if not self.should_sample(frame_idx):
            return self._mode

        candidate = self._classify_frame(frame, face_box)

        # Hysteresis: increment votes for candidate, decay others
        for m in AmbientMode:
            if m == candidate:
                self._votes[m] = min(self._votes[m] + 1, self._hysteresis + 1)
            else:
                self._votes[m] = max(self._votes[m] - 1, 0)

        if self._votes[candidate] >= self._hysteresis:
            self._mode = candidate

        return self._mode

    @property
    def mode(self) -> AmbientMode:
        return self._mode
