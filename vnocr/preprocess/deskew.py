"""Skew estimation & correction (Design §1.2).

Small skew is common in administrative scans and, uncorrected, it costs CER.
Angle is estimated by the projection-variance method: rotate a binarised copy
across a range of angles and pick the angle whose horizontal projection profile
has maximum variance (text rows line up → peaky profile → high variance).  The
grayscale image is then rotated by that angle — binarisation is used only to
*measure* the angle, never to feed the recogniser (§1.1).
"""

from __future__ import annotations

from typing import Tuple

try:
    import cv2
    import numpy as np
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.preprocess needs numpy + opencv-python.") from _e

__all__ = ["estimate_skew", "deskew"]


def _projection_variance(binary: "np.ndarray", angle: float) -> float:
    h, w = binary.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(binary, m, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    proj = rot.sum(axis=1, dtype="float64")
    return float(np.var(proj))


def estimate_skew(gray: "np.ndarray", limit: float = 15.0,
                  step: float = 0.5) -> float:
    """Return the skew angle in degrees within ``±limit`` (coarse→fine)."""
    binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # coarse sweep
    angles = np.arange(-limit, limit + step, step)
    scores = [(_projection_variance(binary, a), a) for a in angles]
    _, best = max(scores)
    # fine sweep around the coarse best
    fine = np.arange(best - step, best + step, step / 5)
    scores = [(_projection_variance(binary, a), a) for a in fine]
    _, best = max(scores)
    return float(best)


def deskew(gray: "np.ndarray", angle: float = None) -> Tuple["np.ndarray", float]:
    """Rotate ``gray`` to remove skew.  Returns ``(deskewed, angle_used)``."""
    if angle is None:
        angle = estimate_skew(gray)
    if abs(angle) < 1e-2:
        return gray, 0.0
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, float(angle)
