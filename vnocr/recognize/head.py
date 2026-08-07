"""Factorised recognition head — Lever 1 (Design §3.2).

Instead of a flat soft-max over ~230 composed classes (which starves ``ữ``,
``ặ``, ``ỡ`` …), the final logit layer is factorised into three additive
factors::

    s(b, m, t | h) = u_b(h) + v_m(h) + w_t(h) + β(b, m, t)
    p(b, m, t | h) = M(b, m, t) · exp(s) / Σ M · exp(s)

* ``u`` base head (~90 classes), ``v`` modifier head (5), ``w`` tone head (6);
* ``β`` a small learned interaction table (the "additivity isn't perfect"
  correction, §3.2) — optional, ablatable;
* ``M`` the validity mask from :class:`vnocr.charset.Charset`, added as ``log M``
  (``-inf`` on impossible glyphs).

Crucially this stays **one CTC alignment**: only the projection factorises.  The
head emits combined per-frame log-probs over ``[blank] + valid_composed`` so a
single :func:`torch.nn.functional.ctc_loss` trains it (Design §3.2).

The auxiliary per-factor logits are returned too, for the flat-vs-factorised
ablation and for tone-only diagnostics (§8).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "vnocr.recognize needs PyTorch. `pip install -e '.[train]'` or "
        "`pip install torch`. The pure-Python core (charset/postprocess/eval) "
        "does not require it."
    ) from _e

from ..charset.charset import Charset, default_charset

__all__ = ["FactorisedHead", "build_flat_index"]

_NEG_INF = -1e9


def build_flat_index(charset: Charset) -> Tuple[List[Tuple[int, int, int]], List[str]]:
    """Enumerate valid ``(b, m, t)`` triples → flat CTC class order.

    Returns ``(triples, chars)`` where ``chars[k]`` is the composed glyph of
    ``triples[k]``.  CTC class 0 is reserved for blank; these occupy 1..K.
    """
    triples: List[Tuple[int, int, int]] = []
    chars: List[str] = []
    from ..charset.decompose import is_valid
    for bi, base in enumerate(charset.bases):
        for mi, mod in enumerate(charset.modifiers):
            for ti, tone in enumerate(charset.tones):
                if is_valid(base, mod, tone):
                    triples.append((bi, mi, ti))
                    chars.append(charset.decode(bi, mi, ti))
    return triples, chars


class FactorisedHead(nn.Module):
    def __init__(self, in_dim: int, charset: Charset = None,
                 use_interaction: bool = True) -> None:
        super().__init__()
        self.charset = charset or default_charset()
        cs = self.charset
        self.n_base, self.n_mod, self.n_tone = cs.n_base, cs.n_mod, cs.n_tone

        # three linear factor heads + a dedicated blank logit
        self.base_proj = nn.Linear(in_dim, self.n_base)
        self.mod_proj = nn.Linear(in_dim, self.n_mod)
        self.tone_proj = nn.Linear(in_dim, self.n_tone)
        self.blank_proj = nn.Linear(in_dim, 1)

        self.use_interaction = use_interaction
        if use_interaction:
            # β(b, m, t): starts at zero so the model begins purely additive.
            self.interaction = nn.Parameter(
                torch.zeros(self.n_base, self.n_mod, self.n_tone))

        # validity mask M(b, m, t) as an additive log-mask buffer
        mask = torch.tensor(cs.build_mask(), dtype=torch.float32)
        log_mask = torch.where(mask > 0, torch.zeros_like(mask),
                               torch.full_like(mask, _NEG_INF))
        self.register_buffer("log_mask", log_mask)  # [B_, M_, T_]

        # flat index: which (b, m, t) cells survive, in CTC class order
        triples, chars = build_flat_index(cs)
        self.flat_chars = chars                    # class k+1 -> glyph
        flat_idx = torch.tensor(triples, dtype=torch.long)  # [K, 3]
        self.register_buffer("flat_b", flat_idx[:, 0])
        self.register_buffer("flat_m", flat_idx[:, 1])
        self.register_buffer("flat_t", flat_idx[:, 2])
        self.num_classes = len(chars) + 1          # + blank at index 0

    def factor_logits(self, feats: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """Per-frame factor logits — used by aux losses and ablation."""
        return {
            "base": self.base_proj(feats),   # [B, T, n_base]
            "mod": self.mod_proj(feats),     # [B, T, n_mod]
            "tone": self.tone_proj(feats),   # [B, T, n_tone]
        }

    def forward(self, feats: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
        """``feats`` ``[B, T, in_dim]`` → combined + factor log-probs.

        Returns
        -------
        dict with:
          ``log_probs`` ``[B, T, K+1]`` — for ``ctc_loss`` (log-softmax, blank=0)
          ``base`` / ``mod`` / ``tone`` — factor logits for aux losses
        """
        B, T, _ = feats.shape
        u = self.base_proj(feats)   # [B, T, n_base]
        v = self.mod_proj(feats)    # [B, T, n_mod]
        w = self.tone_proj(feats)   # [B, T, n_tone]
        blank = self.blank_proj(feats)  # [B, T, 1]

        # additive score s[b,m,t] = u_b + v_m + w_t (+ β) + logM
        s = (u[:, :, :, None, None]
             + v[:, :, None, :, None]
             + w[:, :, None, None, :])
        if self.use_interaction:
            s = s + self.interaction  # broadcast [n_base, n_mod, n_tone]
        s = s + self.log_mask  # zero out impossible glyphs

        # gather the valid combinations into flat CTC classes
        s_flat = s.reshape(B, T, -1)                          # [B, T, B_*M_*T_]
        lin = (self.flat_b * (self.n_mod * self.n_tone)
               + self.flat_m * self.n_tone + self.flat_t)     # [K]
        valid = s_flat.index_select(-1, lin)                  # [B, T, K]

        logits = torch.cat([blank, valid], dim=-1)            # blank at 0
        log_probs = F.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "base": u, "mod": v, "tone": w}
