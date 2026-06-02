import numpy as np
from typing import List, Tuple

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  291, 39,  181, 0,   17, 269, 405]


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _eye_ear(landmarks: List[Tuple[float, float]], indices: List[int]) -> float:
    p = [landmarks[i] for i in indices]
    # EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 × ‖p1−p4‖)
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-6)


def compute_ear(landmarks: List[Tuple[float, float]]) -> float:
    left = _eye_ear(landmarks, LEFT_EYE)
    right = _eye_ear(landmarks, RIGHT_EYE)
    return (left + right) / 2.0


def compute_mar(landmarks: List[Tuple[float, float]]) -> float:
    p = [landmarks[i] for i in MOUTH]
    vertical = (_dist(p[2], p[6]) + _dist(p[3], p[7])) / 2.0
    horizontal = _dist(p[0], p[1]) + 1e-6
    return vertical / horizontal
