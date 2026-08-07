# Data (Design §6)

Where the project is won or lost (§6). Three axes: difficulty, accuracy,
diversity.

## Real datasets (§6.1)
| Dataset | What | Note |
|---|---|---|
| VinText / Vintext (VinAI, CVPR'21) | 2,000 imgs, 56K text entities | 1200/500/300 split |
| 5CD-AI/Viet-Handwriting-OCR (HF) | 23,403 hand-labelled imgs | handwriting |
| HANDS-VNOnDB (ICFHR'18) | overlapping-diacritic eval | stress test |
| Viet-OCR-VQA (Vintern) | 137K+ imgs with Vietnamese text | scale |

Download these yourself (licences vary); drop manifests as
`image_path<TAB>text` for `vnocr.recognize.LineRecognitionDataset`.

## Synthetic (the bulk, §6.2)
```bash
python scripts/build_synth_data.py --corpus corpus.txt --fonts fonts/ \
    --out data/synth --n 100000
```
**Font warning:** many Latin fonts render Vietnamese diacritics wrong. The
generator's `--verify-fonts` renders the stress set (ữ ặ ỡ ẫ …) and drops any
font that can't. Do not skip this — a bad font silently poisons the labels.

## LM corpus (§6.3)
Plain text only, for the n-gram (see `data/lm/README.md`).

## Normalisation
All labels and ground truth go through `vnocr.charset.normalize_text` (NFC +
tone placement, §1.6) before training or scoring. Skipping this inflates CER for
free.
