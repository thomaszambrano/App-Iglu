import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple


class FaceMeshDetector:
    """Thin wrapper around MediaPipe FaceMesh for landmark extraction."""

    def __init__(self, max_faces: int = 1, min_detection_conf: float = 0.5,
                 min_tracking_conf: float = 0.5):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf,
        )

    def process(self, frame_bgr: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Return normalized (x, y) landmarks for the first detected face, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        lms = results.multi_face_landmarks[0].landmark
        h, w = frame_bgr.shape[:2]
        return [(lm.x * w, lm.y * h) for lm in lms]

    def close(self) -> None:
        self._face_mesh.close()
