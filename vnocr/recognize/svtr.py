"""SVTR-style recognition backbone — Lever 2 (Design §3.3).

Two Vietnamese-specific choices distinguish this from a stock Latin STR encoder:

1. **48px input height**, not the 32px inherited from Latin OCR.  Tone marks are
   ~0.10–0.13 em tall; at 32px they fall below the Nyquist limit needed to tell
   ``hỏi`` (one inflection) from ``ngã`` (two).  At 48px they clear it.

2. **Asymmetric, height-preserving stride schedule.**  The tone mark is the
   highest-vertical-frequency component, so every height-halving risks aliasing
   it away.  Early stages therefore use stride ``(1, 2)`` — downsample *width*
   first, keep *height* — and only later stages collapse height, once features
   are abstract.  This is nearly free and is the single cheapest lever (§9).

The encoder is an SVTR-flavoured mix of local (depthwise-conv) and global
(self-attention) mixing blocks between merging stages.  It outputs a sequence
``[B, T, C]`` ready for the factorised CTC head.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.recognize needs PyTorch (`pip install torch`).") from _e

__all__ = ["SVTRBackbone", "svtr_tiny"]


class ConvBNAct(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, groups=1) -> None:
        super().__init__()
        pad = k // 2
        self.conv = nn.Conv2d(cin, cout, k, stride, pad, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LocalMixer(nn.Module):
    """Depthwise conv mixing — cheap local receptive field (SVTR 'local')."""

    def __init__(self, dim, k=(3, 3)) -> None:
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, k, 1, (k[0] // 2, k[1] // 2), groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.act(self.pw(self.norm(self.dw(x))))


class GlobalMixer(nn.Module):
    """Multi-head self-attention over the H×W token grid (SVTR 'global')."""

    def __init__(self, dim, heads=8, mlp_ratio=4.0, drop=0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, dim), nn.Dropout(drop),
        )

    def forward(self, x):  # x: [B, N, C]
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MergingBlock(nn.Module):
    """Down-sampling between stages.  ``stride=(sh, sw)`` sets the asymmetry."""

    def __init__(self, cin, cout, stride: Tuple[int, int]) -> None:
        super().__init__()
        self.conv = ConvBNAct(cin, cout, k=3, stride=stride)

    def forward(self, x):
        return self.conv(x)


class SVTRBackbone(nn.Module):
    """Config-driven SVTR-style encoder with an explicit stride schedule.

    Parameters
    ----------
    in_ch : int
        1 for grayscale (recommended — Design §1.1 keeps grayscale, no binarise),
        3 for colour.
    dims : sequence of int
        Channel width per stage.
    depths : sequence of int
        Number of mixing blocks per stage.
    mixers : sequence of str
        ``"local"`` or ``"global"`` per stage.
    strides : sequence of (sh, sw)
        Height/width stride of the *merge before* each stage.  The default keeps
        height early (``(1, 2)``) and collapses it late — the §3.3 lever.
    """

    def __init__(
        self,
        in_ch: int = 1,
        dims: Sequence[int] = (64, 128, 256),
        depths: Sequence[int] = (2, 2, 4),
        mixers: Sequence[str] = ("local", "local", "global"),
        strides: Sequence[Tuple[int, int]] = ((2, 2), (1, 2), (2, 2)),
        heads: int = 8,
        out_dim: int = 256,
    ) -> None:
        super().__init__()
        assert len(dims) == len(depths) == len(mixers) == len(strides)

        # stem keeps height (stride 1 vertically) — width halved.
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, dims[0] // 2, k=3, stride=(1, 2)),
            ConvBNAct(dims[0] // 2, dims[0], k=3, stride=(1, 1)),
        )

        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        cin = dims[0]
        for i, (dim, depth, mixer, stride) in enumerate(
                zip(dims, depths, mixers, strides)):
            self.merges.append(MergingBlock(cin, dim, stride))
            blocks: List[nn.Module] = []
            for _ in range(depth):
                if mixer == "local":
                    blocks.append(LocalMixer(dim))
                else:
                    blocks.append(_GridAttention(dim, heads))
            self.stages.append(nn.Sequential(*blocks))
            cin = dim

        self.head_norm = nn.LayerNorm(cin)
        self.proj = nn.Linear(cin, out_dim)
        self.out_dim = out_dim

    def forward(self, x):  # x: [B, C, H, W]
        x = self.stem(x)
        for merge, stage in zip(self.merges, self.stages):
            x = merge(x)
            x = stage(x)
        # collapse remaining height by average pooling → sequence over width
        B, C, H, W = x.shape
        x = x.mean(dim=2)             # [B, C, W]
        x = x.transpose(1, 2)        # [B, W, C]
        x = self.proj(self.head_norm(x))
        return x                     # [B, T=W, out_dim]


class _GridAttention(nn.Module):
    """Wrap :class:`GlobalMixer` to accept/return a 4-D feature map."""

    def __init__(self, dim, heads) -> None:
        super().__init__()
        self.mixer = GlobalMixer(dim, heads)

    def forward(self, x):  # [B, C, H, W]
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # [B, H*W, C]
        tokens = self.mixer(tokens)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


def svtr_tiny(in_ch: int = 1, out_dim: int = 256) -> SVTRBackbone:
    """A ~8–12M-param SVTR-tiny with the Vietnamese stride schedule (§2 budget).

    Width is downsampled to **W/4** total (stem ``(1,2)`` × one stage ``(1,2)``),
    so the CTC sequence length ``T = W/4`` comfortably exceeds the character
    count of a line — a harder width reduction (e.g. /16) makes ``T`` shorter
    than the target and CTC fails.  Height is kept at 48 early and collapsed
    **late** via two ``(2,1)`` stages (48→24→12) then mean-pooled, so tone marks
    survive the fewest possible decimations (Lever 2, §3.3).
    """
    return SVTRBackbone(
        in_ch=in_ch,
        dims=(64, 128, 256),
        depths=(2, 2, 4),
        mixers=("local", "local", "global"),
        # merge strides (sh, sw): width first (keep height), then collapse height
        strides=((1, 2), (2, 1), (2, 1)),
        out_dim=out_dim,
    )
