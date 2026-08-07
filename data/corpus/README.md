# Text corpus (Design §6.3)

Plain Vietnamese text, one sentence per line, UTF-8, NFC-normalised. Feeds two
things: the synthetic image text (§6.2) and the tone-recovery LM (§5.3).

## `vi_sample.txt`
A small **original** starter set (~40 sentences, education/administrative
domain). Enough to smoke-test the data + LM plumbing — **not** enough to train a
usable recogniser or LM. Tracked in git on purpose.

## Production corpus
Get real scale (tens of millions of sentences) from:
- Vietnamese **Wikipedia** dump (CC BY-SA) → extract with `wikiextractor`.
- A **news** corpus.

Normalise before use:
```python
from vnocr.charset import normalize_text
open("corpus.txt","w").writelines(
    normalize_text(l) for l in open("corpus.raw", encoding="utf-8"))
```
Large corpora are git-ignored (see `.gitignore`); keep only `vi_sample.txt`.
