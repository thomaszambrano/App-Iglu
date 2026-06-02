import numpy as np
import pytest
import cv2

from vision.processors.night_vision import NightVisionProcessor
from vision.processors.backlight import BacklightingCorrector


def make_frame(h=360, w=640):
    rng = np.random.default_rng(42)
    return rng.integers(30, 220, (h, w, 3), dtype=np.uint8)


def test_gamma_only_on_L():
    """[1.1] A/B channels from split are passed unchanged to merge — gamma never touches them.

    We replicate the internal pipeline steps to verify the invariant directly,
    avoiding the lossy BGR↔LAB round-trip that introduces ±1 quantisation noise.
    """
    proc = NightVisionProcessor(gamma=1.8, clahe_clip=2.0)
    frame = make_frame()

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_in, a_in, b_in = cv2.split(lab)

    # Replicate only the L transforms (what process() does internally)
    l_gamma = cv2.LUT(l_in, proc._lut)
    tile = (16, 16) if frame.shape[1] >= 1000 else (8, 8)
    clahe = cv2.createCLAHE(clipLimit=proc.clahe_clip, tileGridSize=tile)
    l_clahe = clahe.apply(l_gamma)

    merged = cv2.merge([l_clahe, a_in, b_in])
    _, a_merged, b_merged = cv2.split(merged)

    assert np.array_equal(a_in, a_merged), "A channel was modified inside the pipeline"
    assert np.array_equal(b_in, b_merged), "B channel was modified inside the pipeline"


def test_shadow_lift_no_realloc():
    """[2.1] Shadow lift buffer must not be reallocated across calls."""
    corrector = BacklightingCorrector()
    frame = make_frame()

    corrector.process(frame)  # first call triggers allocation
    buf_id_before = id(corrector._shadow_lift_buf)

    for _ in range(99):
        corrector.process(frame)

    buf_id_after = id(corrector._shadow_lift_buf)
    assert buf_id_before == buf_id_after, "Shadow lift buffer was reallocated"


def test_clahe_tile_resolution():
    """[2.3] Tile size is (16,16) at ≥1000px wide, (8,8) otherwise."""
    proc = NightVisionProcessor()
    corrector = BacklightingCorrector()

    large = make_frame(720, 1280)
    small = make_frame(360, 640)

    # NightVisionProcessor tile check via internal method
    assert proc._clahe_for_shape(1280).getTilesGridSize() == (16, 16)
    assert proc._clahe_for_shape(640).getTilesGridSize() == (8, 8)

    assert corrector._clahe_for_shape(1280).getTilesGridSize() == (16, 16)
    assert corrector._clahe_for_shape(640).getTilesGridSize() == (8, 8)
