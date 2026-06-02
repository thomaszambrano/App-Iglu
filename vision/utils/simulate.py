import cv2
import numpy as np


def simulate_night_environment(frame: np.ndarray, factor: float = 0.25) -> np.ndarray:
    """Darken the frame to simulate low-light / night driving conditions."""
    return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def simulate_backlit_environment(frame: np.ndarray, boost: float = 180.0) -> np.ndarray:
    """Add a radial Gaussian background gradient to simulate strong backlighting.

    [5.1] Uses a Gaussian radial gradient (not a hard rectangle) so the
    background boost fades smoothly — creating a light-wrap effect.
    """
    h, w = frame.shape[:2]
    cy, cx = h // 2, w // 2

    # Radial distance from center, normalised to [0, 1]
    ys = np.arange(h)
    xs = np.arange(w)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)

    # Gaussian weight: bright at edges, dark at center (inverted bell)
    sigma = 0.55
    weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    # Invert so edges are bright
    gradient = (1.0 - weight) * boost

    gradient_3ch = np.stack([gradient] * 3, axis=-1).astype(np.float32)
    result = np.clip(frame.astype(np.float32) + gradient_3ch, 0, 255).astype(np.uint8)
    return result
