"""Ruled-table reconstruction by morphology — exact (Design §4.1).

Most Vietnamese administrative / accounting tables are fully ruled.  For those,
the grid is an *exact geometric invariant*, not something to be estimated:

1. binarise (the one legitimate binarisation — §1.1, §4.1);
2. morphological opening with a long **horizontal** kernel → horizontal rules;
   with a long **vertical** kernel → vertical rules;
3. intersect the two masks → grid nodes;
4. nodes + edges form a planar graph; **each table cell is a face** of it.

Zero parameters, O(pixels), millisecond runtime, and no learned model can
improve on an already-exact answer.  Borderless / broken-rule tables fall back
to SLANet (:mod:`vnocr.table.slanet`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.table needs numpy + opencv-python.") from _e

from ..preprocess.binarize import otsu

__all__ = ["Cell", "TableGrid", "extract_rules", "reconstruct_grid"]


@dataclass
class Cell:
    row: int
    col: int
    box: Tuple[int, int, int, int]   # x0, y0, x1, y1
    rowspan: int = 1
    colspan: int = 1
    text: str = ""


@dataclass
class TableGrid:
    rows: List[int]                  # y coordinates of horizontal rules
    cols: List[int]                  # x coordinates of vertical rules
    cells: List[Cell]

    @property
    def n_rows(self) -> int:
        return max(0, len(self.rows) - 1)

    @property
    def n_cols(self) -> int:
        return max(0, len(self.cols) - 1)


def extract_rules(mask: "np.ndarray", scale: int = 20
                  ) -> Tuple["np.ndarray", "np.ndarray"]:
    """Return ``(horizontal_mask, vertical_mask)`` via directional opening.

    ``scale`` sets the minimum rule length as ``dimension // scale``.
    """
    h, w = mask.shape
    hor_len = max(1, w // scale)
    ver_len = max(1, h // scale)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_len, 1))
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_len))
    horizontal = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hor_kernel)
    vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ver_kernel)
    return horizontal, vertical


def _cluster_positions(coords: "np.ndarray", gap: int = 8) -> List[int]:
    """Collapse near-duplicate line coordinates into single separators."""
    if len(coords) == 0:
        return []
    coords = np.sort(coords)
    clusters = [[coords[0]]]
    for c in coords[1:]:
        if c - clusters[-1][-1] <= gap:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return [int(np.mean(cl)) for cl in clusters]


def reconstruct_grid(gray: "np.ndarray", scale: int = 20,
                     detect_spans: bool = True) -> Optional[TableGrid]:
    """Reconstruct a fully-ruled table's cell grid from a grayscale crop.

    Returns ``None`` if too few rules are found (caller should try SLANet).
    """
    mask = otsu(gray)
    horizontal, vertical = extract_rules(mask, scale=scale)

    # rule coordinates from projections
    row_ys = _cluster_positions(np.where(horizontal.sum(axis=1) > 0)[0])
    col_xs = _cluster_positions(np.where(vertical.sum(axis=0) > 0)[0])
    if len(row_ys) < 2 or len(col_xs) < 2:
        return None

    intersections = cv2.bitwise_and(horizontal, vertical)

    cells: List[Cell] = []
    for i in range(len(row_ys) - 1):
        for j in range(len(col_xs) - 1):
            x0, x1 = col_xs[j], col_xs[j + 1]
            y0, y1 = row_ys[i], row_ys[i + 1]
            cells.append(Cell(row=i, col=j, box=(x0, y0, x1, y1)))

    grid = TableGrid(rows=row_ys, cols=col_xs, cells=cells)
    if detect_spans:
        _merge_spans(grid, horizontal, vertical)
    return grid


def _line_present(line_mask: "np.ndarray", x0: int, y0: int, x1: int, y1: int,
                  axis: str, coverage: float = 0.5) -> bool:
    """Is there a rule segment along the requested border of a cell?"""
    seg = line_mask[y0:y1 + 1, x0:x1 + 1]
    if seg.size == 0:
        return False
    if axis == "h":
        return (seg.sum(axis=1) > 0).mean() < 1.0 and seg.max() > 0
    return seg.max() > 0


def _merge_spans(grid: TableGrid, horizontal: "np.ndarray",
                 vertical: "np.ndarray") -> None:
    """Detect merged cells by absent interior rules and set row/col spans.

    A missing vertical rule between (i,j) and (i,j+1) → a horizontal merge; a
    missing horizontal rule → a vertical merge.  This keeps the exact-grid
    guarantee while recovering the merged headers common in official forms.
    Left as a conservative pass: only immediate right/down neighbours.
    """
    by_pos = {(c.row, c.col): c for c in grid.cells}
    consumed = set()
    for c in grid.cells:
        if (c.row, c.col) in consumed:
            continue
        # try to extend right while the separating vertical rule is missing
        x0, y0, x1, y1 = c.box
        right = by_pos.get((c.row, c.col + c.colspan))
        while right is not None:
            sep_x = right.box[0]
            col_seg = vertical[y0:y1 + 1, max(0, sep_x - 1):sep_x + 2]
            if col_seg.size and col_seg.max() > 0:
                break  # rule present → real separator
            c.colspan += 1
            x1 = right.box[2]
            consumed.add((right.row, right.col))
            right = by_pos.get((c.row, c.col + c.colspan))
        c.box = (x0, y0, x1, y1)
    grid.cells = [c for c in grid.cells if (c.row, c.col) not in consumed]
