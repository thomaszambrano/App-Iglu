import numpy as np
import pytest

from vision.pipeline import PreprocessingPipeline
from vision.ambient import AmbientMode
from vision.config import DEFAULT_CONFIG


def make_config(**overrides):
    return {**DEFAULT_CONFIG, **overrides}


def uniform_bgr(luma: int, h=360, w=640) -> np.ndarray:
    return np.full((h, w, 3), max(0, min(255, luma)), dtype=np.uint8)


def test_transition_freeze():
    """[4.1] After a mode change, skip_inference=True for exactly transition_freeze_frames."""
    freeze = 5
    cfg = make_config(
        transition_freeze_frames=freeze,
        night_vision_trigger_luma=55,
        hysteresis_buffer=1,  # instant mode change for test
    )
    pipeline = PreprocessingPipeline(cfg)

    # Force internal mode state + trigger transition
    pipeline._current_mode = AmbientMode.NORMAL
    pipeline._ambient._mode = AmbientMode.NIGHT_VISION
    pipeline._ambient._votes[AmbientMode.NIGHT_VISION] = 2

    # Dark frame will cause ambient to return NIGHT_VISION
    dark = uniform_bgr(20)
    results = [pipeline.process(dark) for _ in range(freeze + 3)]

    skip_values = [r["skip_inference"] for r in results]
    # The first `freeze` frames after mode change should have skip_inference=True
    true_count = sum(1 for v in skip_values if v)
    assert true_count == freeze, f"Expected {freeze} frozen frames, got {true_count}"


def test_internal_counter():
    """[4.2] After 50 calls to process(), _frame_idx == 50."""
    pipeline = PreprocessingPipeline()
    frame = uniform_bgr(128)
    for _ in range(50):
        pipeline.process(frame)
    assert pipeline._frame_idx == 50


def test_no_flicker_at_boundary():
    """[5.2] 1800 frames at exactly night_vision_trigger_luma → fewer than 2 mode flips."""
    trigger = 55
    cfg = make_config(
        night_vision_trigger_luma=trigger,
        hysteresis_buffer=8,
        transition_freeze_frames=5,
    )
    pipeline = PreprocessingPipeline(cfg)

    # Frame whose LAB L channel ≈ trigger value
    # LAB L range is 0-255 in OpenCV; set pixel value to trigger
    boundary_frame = uniform_bgr(trigger)

    mode_history = []
    for _ in range(1800):
        result = pipeline.process(boundary_frame)
        mode_history.append(result["mode"])

    flips = sum(1 for i in range(1, len(mode_history)) if mode_history[i] != mode_history[i - 1])
    assert flips < 2, f"Mode flipped {flips} times — hysteresis not working"
