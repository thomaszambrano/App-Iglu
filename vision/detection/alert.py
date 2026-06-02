from ..ambient import AmbientMode

# Per-mode EAR offset applied to lower the effective threshold
_MODE_EAR_OFFSET: dict[AmbientMode, float] = {
    AmbientMode.NIGHT_VISION: -0.02,
    AmbientMode.BACKLIT: -0.01,
    AmbientMode.NORMAL: 0.0,
}


class AlertManager:
    """Tracks EAR/MAR over time and fires drowsiness alerts with mode-aware thresholds."""

    def __init__(self, ear_threshold: float = 0.25, mar_threshold: float = 0.6,
                 consec_frames: int = 20):
        self.base_ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self._ear_counter = 0
        self._mar_counter = 0
        self.drowsy = False
        self.yawning = False

    def effective_ear_threshold(self, mode: AmbientMode) -> float:
        return self.base_ear_threshold + _MODE_EAR_OFFSET[mode]

    def update(self, ear: float, mar: float, mode: AmbientMode) -> dict:
        ear_thresh = self.effective_ear_threshold(mode)

        if ear < ear_thresh:
            self._ear_counter += 1
        else:
            self._ear_counter = max(0, self._ear_counter - 1)

        if mar > self.mar_threshold:
            self._mar_counter += 1
        else:
            self._mar_counter = max(0, self._mar_counter - 1)

        self.drowsy = self._ear_counter >= self.consec_frames
        self.yawning = self._mar_counter >= self.consec_frames

        return {
            "drowsy": self.drowsy,
            "yawning": self.yawning,
            "ear": ear,
            "mar": mar,
            "ear_threshold": ear_thresh,
            "mode": mode.name,
        }
