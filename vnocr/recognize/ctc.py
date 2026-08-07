"""CTC loss and decoding for the factorised head (Design §3.2).

The head emits a single log-prob distribution over ``[blank] + valid_composed``,
so training is one ordinary ``ctc_loss`` — the factorisation does not add CTC
alignments.  Greedy decoding here maps flat class ids back to composed glyphs
via the head's ``flat_chars``; the dictionary-constrained beam search that uses
the full lattice lives in :mod:`vnocr.postprocess.decode`.
"""

from __future__ import annotations

from typing import List, Sequence

try:
    import torch
    import torch.nn.functional as F
except ImportError as _e:  # pragma: no cover
    raise ImportError("vnocr.recognize needs PyTorch (`pip install torch`).") from _e

__all__ = ["ctc_loss", "greedy_decode", "flat_log_probs_to_lattice"]


def ctc_loss(log_probs: "torch.Tensor", targets: "torch.Tensor",
             input_lengths: "torch.Tensor", target_lengths: "torch.Tensor",
             blank: int = 0) -> "torch.Tensor":
    """Thin wrapper: ``log_probs`` is ``[B, T, C]`` (blank at 0).

    Internally transposed to the ``[T, B, C]`` that ``F.ctc_loss`` expects.
    """
    lp = log_probs.permute(1, 0, 2)  # [T, B, C]
    return F.ctc_loss(lp, targets, input_lengths, target_lengths,
                      blank=blank, zero_infinity=True)


def greedy_decode(log_probs: "torch.Tensor", flat_chars: Sequence[str],
                  blank: int = 0) -> List[str]:
    """Best-path CTC decode → list of strings (one per batch item).

    ``flat_chars[k]`` is the glyph of class ``k + 1`` (class 0 is blank).
    """
    best = log_probs.argmax(dim=-1)  # [B, T]
    out: List[str] = []
    for row in best.tolist():
        chars: List[str] = []
        prev = -1
        for cls in row:
            if cls != prev and cls != blank:
                chars.append(flat_chars[cls - 1])
            prev = cls
        out.append("".join(chars))
    return out


def flat_log_probs_to_lattice(log_probs: "torch.Tensor") -> List[List[List[float]]]:
    """Detach a ``[B, T, C]`` tensor to nested Python lists.

    Bridges the neural recogniser to the pure-Python
    :func:`vnocr.postprocess.decode.ctc_prefix_beam_search`, which takes a
    ``[T][C]`` list per sample so it runs without numpy on the serving box.
    """
    return log_probs.detach().cpu().tolist()
