"""End-to-end OCR orchestration (Design §2).

    image → preprocess → layout ─┬─ text  → detect → recognise → post-process
                                 └─ table → structure → per-cell recognise
                                                    ↓
                                          merge → text / JSON / HTML

The pipeline is **modular and non-generative**: because no block emits language,
it cannot hallucinate (unlike an end-to-end VLM); every box is exact and every
block is independently swappable/debuggable (§2).

Components are injected, not hard-wired.  The classical blocks (preprocess,
layout, table morphology) run with just numpy/opencv; recognition needs a loaded
:class:`~vnocr.recognize.model.VietRecognizer` (torch).  Whatever you don't pass
is simply skipped, so you can exercise the pipeline incrementally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from .charset.normalize import normalize_text

__all__ = ["OCRResult", "OCRPipeline"]


@dataclass
class TextLine:
    box: Any
    text: str


@dataclass
class TableResult:
    box: Any
    grid: Any                    # vnocr.table.TableGrid


@dataclass
class OCRResult:
    lines: List[TextLine] = field(default_factory=list)
    tables: List[TableResult] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)

    def to_json(self) -> dict:
        return {
            "lines": [{"box": l.box, "text": l.text} for l in self.lines],
            "tables": [
                {"box": t.box,
                 "cells": [{"row": c.row, "col": c.col, "box": c.box,
                            "rowspan": c.rowspan, "colspan": c.colspan,
                            "text": c.text} for c in t.grid.cells]}
                for t in self.tables
            ],
        }

    def to_html(self) -> str:
        parts = [f"<p>{_esc(l.text)}</p>" for l in self.lines]
        for t in self.tables:
            parts.append(_grid_to_html(t.grid))
        return "\n".join(parts)


class OCRPipeline:
    def __init__(self, preprocessor=None, layout=None, detector=None,
                 recognizer=None, postprocessor=None, table_reconstructor=None,
                 constrained_decode: bool = True) -> None:
        self.preprocessor = preprocessor
        self.layout = layout
        self.detector = detector
        self.recognizer = recognizer
        self.postprocessor = postprocessor
        # callable(gray) -> TableGrid or None; defaults to morphology
        self.table_reconstructor = table_reconstructor
        self.constrained_decode = constrained_decode

    # -- public API -------------------------------------------------------
    def run(self, image) -> OCRResult:
        """Process a page image (numpy array) into an :class:`OCRResult`."""
        gray = self.preprocessor(image) if self.preprocessor else image
        result = OCRResult()

        regions = self.layout.analyze(gray) if self.layout else None
        if regions is None:
            # no layout model: treat the whole page as one text region
            self._run_text_region(gray, None, result)
            return result

        from .layout import RegionType
        for region in regions:
            x0, y0, x1, y1 = region.box
            crop = gray[y0:y1, x0:x1]
            if region.type == RegionType.TABLE:
                self._run_table_region(crop, region.box, result)
            else:
                self._run_text_region(crop, region.box, result)
        return result

    # -- branches ---------------------------------------------------------
    def _run_text_region(self, crop, box, result: OCRResult) -> None:
        if self.recognizer is None:
            return
        line_crops, line_boxes = self._detect_lines(crop, box)
        for lc, lb in zip(line_crops, line_boxes):
            text = self._recognize_line(lc)
            if self.postprocessor is not None:
                text = self.postprocessor.run(text)
            result.lines.append(TextLine(box=lb, text=normalize_text(text)))

    def _run_table_region(self, crop, box, result: OCRResult) -> None:
        reconstruct = self.table_reconstructor
        if reconstruct is None:
            from .table import reconstruct_grid
            reconstruct = reconstruct_grid
        grid = reconstruct(crop)
        if grid is None:
            # borderless → SLANet path would go here; fall back to text branch
            self._run_text_region(crop, box, result)
            return
        if self.recognizer is not None:
            for cell in grid.cells:
                cx0, cy0, cx1, cy1 = cell.box
                cell_img = crop[cy0:cy1, cx0:cx1]
                txt = self._recognize_line(cell_img)
                if self.postprocessor is not None:
                    txt = self.postprocessor.run(txt)
                cell.text = normalize_text(txt)
        result.tables.append(TableResult(box=box, grid=grid))

    # -- helpers ----------------------------------------------------------
    def _detect_lines(self, crop, box):
        """Return (line_crops, line_boxes).  Without a detector, use the crop."""
        if self.detector is None:
            return [crop], [box]
        return self.detector(crop)   # user-supplied callable → (crops, boxes)

    def _recognize_line(self, line_img) -> str:
        rec = self.recognizer
        tensor = _to_line_tensor(line_img)
        out = rec(tensor)
        if self.constrained_decode and self.postprocessor is not None \
                and getattr(self.postprocessor, "trie", None) is not None:
            from .recognize.ctc import flat_log_probs_to_lattice
            from .postprocess.decode import ctc_prefix_beam_search
            lattice = flat_log_probs_to_lattice(out["log_probs"])[0]
            alphabet = ["<blank>"] + list(rec.flat_chars)
            lm = getattr(self.postprocessor.tone, "lm", None) \
                if self.postprocessor.tone else None
            return ctc_prefix_beam_search(
                lattice, alphabet, blank=0,
                trie=self.postprocessor.trie, lm=lm)
        return rec.decode(out["log_probs"])[0]


def _to_line_tensor(line_img):
    """Grayscale line image (numpy) → [1,1,48,W] float tensor in [0,1]."""
    import numpy as np
    import torch
    from .recognize.dataset import IMG_HEIGHT
    if line_img.ndim == 3:
        import cv2
        line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    h, w = line_img.shape[:2]
    new_w = max(1, round(w * IMG_HEIGHT / h))
    import cv2
    resized = cv2.resize(line_img, (new_w, IMG_HEIGHT))
    arr = resized.astype("float32") / 255.0
    return torch.from_numpy(arr)[None, None, :, :]


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _grid_to_html(grid) -> str:
    rows: dict = {}
    for c in grid.cells:
        rows.setdefault(c.row, []).append(c)
    out = ["<table border=\"1\">"]
    for r in sorted(rows):
        out.append("<tr>")
        for c in sorted(rows[r], key=lambda x: x.col):
            span = ""
            if c.colspan > 1:
                span += f" colspan=\"{c.colspan}\""
            if c.rowspan > 1:
                span += f" rowspan=\"{c.rowspan}\""
            out.append(f"<td{span}>{_esc(c.text)}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)
