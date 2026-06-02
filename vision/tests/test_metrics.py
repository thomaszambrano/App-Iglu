import numpy as np
import pytest

from vision.detection.alert import AlertManager
from vision.processors.night_vision import NightVisionProcessor
from vision.ambient import AmbientMode


def test_night_vision_ear_offset():
    """[4.3] NIGHT_VISION effective threshold is lower than NORMAL by exactly 0.02."""
    mgr = AlertManager(ear_threshold=0.25)
    normal_thresh = mgr.effective_ear_threshold(AmbientMode.NORMAL)
    night_thresh = mgr.effective_ear_threshold(AmbientMode.NIGHT_VISION)
    assert abs(normal_thresh - night_thresh - 0.02) < 1e-9, (
        f"Expected 0.02 offset, got {normal_thresh - night_thresh:.4f}"
    )


def test_backlit_ear_offset():
    """[4.3] BACKLIT effective threshold is lower than NORMAL by exactly 0.01."""
    mgr = AlertManager(ear_threshold=0.25)
    normal_thresh = mgr.effective_ear_threshold(AmbientMode.NORMAL)
    backlit_thresh = mgr.effective_ear_threshold(AmbientMode.BACKLIT)
    assert abs(normal_thresh - backlit_thresh - 0.01) < 1e-9


def test_reconfigure_rebuilds_lut():
    """[4.4] After reconfigure(2.0), LUT[128] differs from LUT built with gamma=1.8."""
    proc = NightVisionProcessor(gamma=1.8)
    lut_before = proc._lut[128]
    proc.reconfigure(2.0)
    lut_after = proc._lut[128]
    assert lut_before != lut_after, (
        f"LUT[128] unchanged after reconfigure: {lut_before} == {lut_after}"
    )
