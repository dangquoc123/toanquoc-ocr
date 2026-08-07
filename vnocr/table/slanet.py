"""SLANet — borderless / broken-rule table structure (Design §4.2).

When too few rules are found for the morphological reconstructor (§4.1), fall
back to SLANet: a ~1M-param CNN + attention decoder that emits an HTML structure
token stream (``<tr>``, ``<td>``, ``<td colspan=…>`` …) plus per-cell bbox
regression.  It is only ~0.41% S-TEDS behind UniTable-Large on PubTabNet while
running on CPU (§4.2).

Table structure is **language-independent** (§4.2): pretrain on PubTabNet /
SynthTabNet (500k+ free tables), then fine-tune on a few thousand Vietnamese
tables — no large-scale structure labelling needed.

This is a compact, faithful architecture skeleton (PyTorch, optional dep); wire
your PubTabNet dataloader to :mod:`scripts.train_recognizer`-style training.
"""

from __future__ import annotations

from typing import Dict, List

try:
    import torch
    import torch.nn as nn
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.table.slanet needs PyTorch (`pip install torch`).") from _e

__all__ = ["SLANet", "STRUCTURE_TOKENS"]

# Minimal HTML structure vocabulary for the token decoder.
STRUCTURE_TOKENS: List[str] = [
    "<pad>", "<s>", "</s>",
    "<table>", "</table>", "<tr>", "</tr>",
    "<td>", "</td>", "<td", ">", " colspan=\"2\"", " colspan=\"3\"",
    " rowspan=\"2\"", " rowspan=\"3\"", "<td></td>",
]


class _PPLCNet(nn.Module):
    """Tiny PP-LCNet-ish CNN encoder (~1M params) — the SLANet backbone."""

    def __init__(self, in_ch: int = 3, width: int = 64) -> None:
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                nn.BatchNorm2d(cin), nn.Hardswish(),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout), nn.Hardswish(),
            )
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(width), nn.Hardswish())
        self.stages = nn.Sequential(
            block(width, width * 2, 2),
            block(width * 2, width * 4, 2),
            block(width * 4, width * 4, 1),
        )
        self.out_dim = width * 4

    def forward(self, x):
        return self.stages(self.stem(x))   # [B, C, H, W]


class SLANet(nn.Module):
    """Structure-attention decoder producing HTML tokens + cell boxes."""

    def __init__(self, n_tokens: int = len(STRUCTURE_TOKENS),
                 dim: int = 256, heads: int = 8, layers: int = 2,
                 max_len: int = 600, in_ch: int = 3) -> None:
        super().__init__()
        self.backbone = _PPLCNet(in_ch=in_ch)
        self.enc_proj = nn.Linear(self.backbone.out_dim, dim)
        self.embed = nn.Embedding(n_tokens, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.token_head = nn.Linear(dim, n_tokens)   # structure token logits
        self.bbox_head = nn.Linear(dim, 4)            # per-token cell bbox (x,y,w,h)
        self.max_len = max_len

    def encode(self, image: "torch.Tensor") -> "torch.Tensor":
        feat = self.backbone(image)                  # [B, C, H, W]
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)     # [B, HW, C]
        return self.enc_proj(tokens)

    def forward(self, image: "torch.Tensor",
                tgt_in: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        mem = self.encode(image)
        L = tgt_in.size(1)
        y = self.embed(tgt_in) + self.pos[:, :L]
        causal = torch.triu(torch.full((L, L), float("-inf"), device=y.device), 1)
        h = self.decoder(y, mem, tgt_mask=causal)
        return {"tokens": self.token_head(h), "bboxes": self.bbox_head(h).sigmoid()}
