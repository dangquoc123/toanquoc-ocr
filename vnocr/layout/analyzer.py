"""Page layout analysis (Design §2).

Splits a page into regions and routes each to the right branch: text regions to
detect→recognise→post-process, table regions to the table reconstructor.  The
design's neural option is PP-DocLayout-lite / YOLO-doc (~5M params); a learned
detector plugs into :meth:`LayoutAnalyzer.analyze` via ``model``.

The default heuristic needs no model and no torch: it finds ruled rectangular
regions (long horizontal + vertical rules) and labels them ``table``; the rest
of the page is ``text``.  Good enough to drive the two-branch pipeline on clean
administrative scans, and a sensible fallback when no layout model is loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

try:
    import cv2
    import numpy as np
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.layout needs numpy + opencv-python.") from _e

from ..preprocess.binarize import otsu

__all__ = ["RegionType", "Region", "LayoutAnalyzer"]


class RegionType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    TITLE = "title"


@dataclass
class Region:
    type: RegionType
    box: tuple           # x0, y0, x1, y1


class LayoutAnalyzer:
    def __init__(self, model=None, min_table_area_frac: float = 0.02) -> None:
        self.model = model
        self.min_table_area_frac = min_table_area_frac

    def analyze(self, gray: "np.ndarray") -> List[Region]:
        if self.model is not None:
            return self._analyze_model(gray)
        return self._analyze_heuristic(gray)

    def _analyze_model(self, gray):  # pragma: no cover - requires a trained model
        """Run a learned layout detector.  Expected to return list[Region]."""
        return self.model.predict(gray)

    def _analyze_heuristic(self, gray: "np.ndarray") -> List[Region]:
        h, w = gray.shape[:2]
        mask = otsu(gray)
        hor = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (max(1, w // 15), 1)))
        ver = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // 15))))
        grid = cv2.dilate(cv2.bitwise_or(hor, ver),
                          cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        regions: List[Region] = []
        table_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.min_table_area_frac * h * w
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw * ch < min_area:
                continue
            # a table region has both long horizontal and vertical rules inside
            sub_h = hor[y:y + ch, x:x + cw].max()
            sub_v = ver[y:y + ch, x:x + cw].max()
            if sub_h > 0 and sub_v > 0:
                regions.append(Region(RegionType.TABLE, (x, y, x + cw, y + ch)))
                table_mask[y:y + ch, x:x + cw] = 255

        # everything not claimed by a table is treated as one text region per
        # connected text block (coarse: the whole page minus tables).
        text_area = cv2.bitwise_and(mask, cv2.bitwise_not(table_mask))
        if text_area.max() > 0:
            ys, xs = np.where(text_area > 0)
            regions.append(Region(RegionType.TEXT,
                                  (int(xs.min()), int(ys.min()),
                                   int(xs.max()), int(ys.max()))))
        return regions
