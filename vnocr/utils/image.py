"""Small image helpers (optional numpy/opencv)."""

from __future__ import annotations

from typing import List, Tuple

__all__ = ["imread", "crop", "order_boxes_reading"]


def imread(path: str):
    """Read an image as a numpy array (BGR or gray).  Needs opencv."""
    import cv2
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    return im


def crop(image, box: Tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    return image[y0:y1, x0:x1]


def order_boxes_reading(boxes: List[Tuple[int, int, int, int]],
                        line_tol: int = 10) -> List[int]:
    """Return indices of ``boxes`` in natural reading order (top→bottom, L→R)."""
    idx = list(range(len(boxes)))
    idx.sort(key=lambda i: (boxes[i][1], boxes[i][0]))
    # group into lines by y proximity, then sort each line by x
    lines: List[List[int]] = []
    for i in idx:
        placed = False
        for line in lines:
            if abs(boxes[line[0]][1] - boxes[i][1]) <= line_tol:
                line.append(i)
                placed = True
                break
        if not placed:
            lines.append([i])
    out: List[int] = []
    for line in lines:
        line.sort(key=lambda i: boxes[i][0])
        out.extend(line)
    return out
