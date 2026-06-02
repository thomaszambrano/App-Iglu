import numpy as np
from typing import Optional, Tuple

from .config import DEFAULT_CONFIG
from .ambient import AmbientDetector, AmbientMode
from .processors.night_vision import NightVisionProcessor
from .processors.backlight import BacklightingCorrector
from .utils.blur import BlurCheck
from .utils.timing import ComponentTimer


class PreprocessingPipeline:
    """Orchestrates ambient detection → preprocessing → frame delivery."""

    def __init__(self, config: Optional[dict] = None):
        self._config = config or DEFAULT_CONFIG
        self._ambient = AmbientDetector(self._config)
        self._night = NightVisionProcessor(
            gamma=self._config["gamma"],
            clahe_clip=self._config["clahe_clip"],
        )
        self._backlight = BacklightingCorrector(clahe_clip=self._config["clahe_clip"])
        self._blur = BlurCheck(threshold=self._config["blur_threshold"])

        # [4.2] Pipeline owns the frame counter — no external frame_index argument
        self._frame_idx: int = 0
        # [4.1] Transition freeze wired in
        self._transition_remaining: int = 0
        self._current_mode: AmbientMode = AmbientMode.NORMAL

    def process(
        self, frame: np.ndarray,
        face_box: Optional[Tuple[int, int, int, int]] = None
    ) -> dict:
        """Process one frame. Returns preprocessed frame + metadata dict."""
        with ComponentTimer("ambient") as t_ambient:
            new_mode = self._ambient.update(frame, self._frame_idx, face_box)

        # [4.1] Detect mode change and start freeze
        if new_mode != self._current_mode:
            self._current_mode = new_mode
            self._transition_remaining = self._config["transition_freeze_frames"]

        skip_inference = self._transition_remaining > 0
        if self._transition_remaining > 0:
            self._transition_remaining -= 1

        blurry = self._blur.is_blurry(frame)

        with ComponentTimer("preprocess") as t_pre:
            if self._current_mode == AmbientMode.NIGHT_VISION:
                processed = self._night.process(frame)
            elif self._current_mode == AmbientMode.BACKLIT:
                processed = self._backlight.process(frame)
            else:
                processed = frame

        self._frame_idx += 1

        return {
            "frame": processed,
            "mode": self._current_mode,
            "skip_inference": skip_inference,
            "blurry": blurry,
            "ambient_ms": t_ambient.elapsed_ms,
            "preprocess_ms": t_pre.elapsed_ms,
            "frame_idx": self._frame_idx,
        }
