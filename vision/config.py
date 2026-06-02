DEFAULT_CONFIG = {
    "night_vision_trigger_luma": 55,
    "backlit_trigger_ratio": 2.4,
    "hysteresis_buffer": 8,
    "blur_threshold": 80.0,
    "gamma": 1.8,
    "clahe_clip": 2.0,
    "transition_freeze_frames": 5,
}

LOCAL_TEST_CONFIG = {
    **DEFAULT_CONFIG,
    "night_vision_trigger_luma": 100,
    "backlit_trigger_ratio": 1.6,
    "blur_threshold": 60.0,
}
