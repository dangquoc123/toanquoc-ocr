"""Line-recognition dataset and label codec (Design §3, §6).

* :class:`LabelCodec` maps a normalised string to the flat CTC class ids the
  factorised head uses, and to the per-factor (base/mod/tone) ids for the
  optional auxiliary heads.  It also normalises labels (NFC + tone placement,
  §1.6) so training targets are canonical.
* :class:`LineRecognitionDataset` reads a ``path<TAB>text`` manifest, loads each
  crop as **grayscale** (Design §1.1 — no hard binarisation), resizes to height
  48 (Lever 2, §3.3) keeping aspect ratio, and returns tensors.
* :func:`collate_fn` pads a batch to equal width and builds the length tensors
  ``ctc_loss`` needs.

Image IO uses Pillow if present, else OpenCV; both are optional and only needed
for actual training/inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..charset.charset import Charset, default_charset
from ..charset.decompose import is_valid
from ..charset.normalize import normalize_text

__all__ = ["LabelCodec", "LineRecognitionDataset", "collate_fn", "IMG_HEIGHT"]

IMG_HEIGHT = 48  # Lever 2 (§3.3)


class LabelCodec:
    """Text ↔ flat CTC class ids for the factorised head.

    Class 0 is blank; valid ``(base, mod, tone)`` triples occupy 1..K in the
    same order as :func:`vnocr.recognize.head.build_flat_index`.
    """

    def __init__(self, charset: Optional[Charset] = None) -> None:
        self.charset = charset or default_charset()
        self._triple_to_cls: Dict[Tuple[int, int, int], int] = {}
        self.chars: List[str] = []
        k = 1  # 0 reserved for blank
        cs = self.charset
        for bi, base in enumerate(cs.bases):
            for mi, mod in enumerate(cs.modifiers):
                for ti, tone in enumerate(cs.tones):
                    if is_valid(base, mod, tone):
                        self._triple_to_cls[(bi, mi, ti)] = k
                        self.chars.append(cs.decode(bi, mi, ti))
                        k += 1
        self.num_classes = k  # includes blank

    def encode(self, text: str, normalize: bool = True) -> List[int]:
        if normalize:
            text = normalize_text(text)
        ids: List[int] = []
        for ch in text:
            try:
                b, m, t = self.charset.encode(ch)
            except KeyError:
                continue  # skip chars outside the inventory
            cls = self._triple_to_cls.get((b, m, t))
            if cls is not None:
                ids.append(cls)
        return ids

    def encode_factors(self, text: str, normalize: bool = True
                       ) -> Tuple[List[int], List[int], List[int]]:
        """Return parallel (base, mod, tone) id lists for the aux heads (§7.2)."""
        if normalize:
            text = normalize_text(text)
        bs, ms, ts = [], [], []
        for ch in text:
            try:
                b, m, t = self.charset.encode(ch)
            except KeyError:
                continue
            if (b, m, t) in self._triple_to_cls:
                bs.append(b)
                ms.append(m)
                ts.append(t)
        return bs, ms, ts

    def decode(self, ids: Sequence[int]) -> str:
        out = []
        for i in ids:
            if 1 <= i < self.num_classes:
                out.append(self.chars[i - 1])
        return "".join(out)


def _load_grayscale(path: str, height: int = IMG_HEIGHT):
    """Load an image as a float32 grayscale array in [0, 1], resized to ``height``."""
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
        img = Image.open(path).convert("L")
        w = max(1, round(img.width * height / img.height))
        img = img.resize((w, height))
        return (np.asarray(img, dtype="float32") / 255.0)[None, :, :]  # [1,H,W]
    except ImportError:
        pass
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    w = max(1, round(im.shape[1] * height / im.shape[0]))
    im = cv2.resize(im, (w, height))
    return (im.astype("float32") / 255.0)[None, :, :]


@dataclass
class Sample:
    image_path: str
    text: str


class LineRecognitionDataset:
    """A ``torch.utils.data.Dataset`` over a ``path<TAB>text`` manifest.

    Subclasses :class:`torch.utils.data.Dataset` lazily so importing this module
    doesn't require torch until you actually build a dataset.
    """

    def __init__(self, manifest: str, codec: Optional[LabelCodec] = None,
                 root: str = "", height: int = IMG_HEIGHT) -> None:
        self.codec = codec or LabelCodec()
        self.height = height
        self.root = Path(root)
        self.samples: List[Sample] = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or "\t" not in line:
                    continue
                p, text = line.split("\t", 1)
                self.samples.append(Sample(p, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        import torch
        s = self.samples[idx]
        arr = _load_grayscale(str(self.root / s.image_path), self.height)
        image = torch.from_numpy(arr)
        target = torch.tensor(self.codec.encode(s.text), dtype=torch.long)
        return {"image": image, "target": target, "text": normalize_text(s.text)}


def collate_fn(batch):
    """Pad grayscale line images to equal width; build CTC + teacher tensors.

    Returns both the **concatenated** targets ``ctc_loss`` wants and a
    **padded** ``[B, Lmax]`` matrix (pad = 0 = blank/PAD) for the optional GTC
    seq2seq teacher (§3.1).
    """
    import torch
    heights = {b["image"].shape[1] for b in batch}
    assert len(heights) == 1, "all crops must share the fixed height"
    max_w = max(b["image"].shape[2] for b in batch)
    imgs = []
    for b in batch:
        img = b["image"]
        pad = max_w - img.shape[2]
        if pad:
            img = torch.nn.functional.pad(img, (0, pad), value=1.0)  # white pad
        imgs.append(img)
    images = torch.stack(imgs)                                  # [B,1,H,Wmax]
    targets = torch.cat([b["target"] for b in batch])          # concat (CTC)
    target_lengths = torch.tensor([len(b["target"]) for b in batch])

    # padded per-sample targets for the teacher (0 = PAD/blank)
    max_l = max(1, int(target_lengths.max().item()) if len(batch) else 1)
    padded = torch.zeros(len(batch), max_l, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["target"]
        if len(t):
            padded[i, :len(t)] = t
    return {
        "images": images,
        "targets": targets,
        "target_lengths": target_lengths,
        "seq_targets": padded,          # [B, Lmax], pad=0
        "texts": [b["text"] for b in batch],
    }
