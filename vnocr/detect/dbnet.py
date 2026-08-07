"""DBNet text detection (Design §2 budget, §3).

Differentiable-Binarisation detector: a light backbone (LCNet-style) + FPN neck
+ a head predicting a probability map and a threshold map, combined by the DB
approximate step function::

    B̂ = 1 / (1 + exp(-k · (P − T)))

Training supervises P, T and B̂; inference thresholds P (or B̂) and extracts
polygons.  Kept in the §2 budget (3–5M params, ~1.5MB INT8).

Box extraction (:func:`boxes_from_prob`) uses OpenCV contours + unclipping and
needs numpy/opencv; the network needs torch.  Both optional.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.detect needs PyTorch (`pip install torch`).") from _e

__all__ = ["DBNet", "boxes_from_prob"]


class _ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _LCNet(nn.Module):
    """Compact backbone returning a 4-level feature pyramid (C2..C5)."""

    def __init__(self, in_ch=3, width=32):
        super().__init__()
        self.stem = _ConvBNAct(in_ch, width, 3, 2)
        self.c2 = nn.Sequential(_ConvBNAct(width, width * 2, 3, 2, g=1))
        self.c3 = nn.Sequential(_ConvBNAct(width * 2, width * 4, 3, 2))
        self.c4 = nn.Sequential(_ConvBNAct(width * 4, width * 8, 3, 2))
        self.c5 = nn.Sequential(_ConvBNAct(width * 8, width * 8, 3, 2))

    def forward(self, x):
        x = self.stem(x)
        c2 = self.c2(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return c2, c3, c4, c5


class _FPN(nn.Module):
    """Feature-pyramid neck (RepLKFPN-style; plain FPN skeleton here)."""

    def __init__(self, channels: Tuple[int, ...], out: int = 64):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, out, 1) for c in channels])
        self.smooth = nn.ModuleList([_ConvBNAct(out, out, 3) for _ in channels])
        self.out_dim = out * len(channels)

    def forward(self, feats):
        laterals = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
        outs = [s(l) for s, l in zip(self.smooth, laterals)]
        size = outs[0].shape[-2:]
        outs = [F.interpolate(o, size=size, mode="nearest") for o in outs]
        return torch.cat(outs, dim=1)


class DBNet(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 32, k: float = 50.0):
        super().__init__()
        self.backbone = _LCNet(in_ch, width)
        chans = (width * 2, width * 4, width * 8, width * 8)
        self.neck = _FPN(chans, out=64)
        self.k = k
        self.prob_head = self._head(self.neck.out_dim)
        self.thresh_head = self._head(self.neck.out_dim)

    @staticmethod
    def _head(cin):
        return nn.Sequential(
            _ConvBNAct(cin, cin // 4, 3),
            nn.ConvTranspose2d(cin // 4, cin // 4, 2, 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(cin // 4, 1, 2, 2), nn.Sigmoid(),
        )

    def forward(self, x) -> Dict[str, "torch.Tensor"]:
        feats = self.neck(self.backbone(x))
        prob = self.prob_head(feats)
        out = {"prob": prob}
        if self.training:
            thresh = self.thresh_head(feats)
            binary = 1.0 / (1.0 + torch.exp(-self.k * (prob - thresh)))
            out.update({"thresh": thresh, "binary": binary})
        return out


def boxes_from_prob(prob_map, box_thresh: float = 0.3, min_size: int = 3,
                    unclip_ratio: float = 1.6) -> List["list"]:
    """Extract quadrilateral text boxes from a probability map.

    Uses OpenCV contour detection + polygon unclipping.  ``prob_map`` is a 2-D
    numpy array in [0, 1].
    """
    import cv2
    import numpy as np

    mask = (prob_map > box_thresh).astype("uint8")
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[list] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_size * min_size:
            continue
        rect = cv2.minAreaRect(cnt)
        (w, h) = rect[1]
        if min(w, h) < min_size:
            continue
        # unclip: expand the box outward by unclip_ratio
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, True)
        distance = area * unclip_ratio / max(length, 1e-6)
        box = cv2.boxPoints(rect)
        center = box.mean(axis=0)
        box = box + (box - center) / (np.linalg.norm(box - center, axis=1,
                                                     keepdims=True) + 1e-6) * distance
        boxes.append(box.astype("int32").tolist())
    return boxes
