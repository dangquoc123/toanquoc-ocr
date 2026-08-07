# Train trên Kaggle GPU free — hướng dẫn từng bước

Kaggle cho **30 giờ GPU/tuần** miễn phí (T4 x2 hoặc P100), mỗi phiên tối đa
12 giờ — đủ cho các giai đoạn đầu (§7.1). Notebook bấm-chạy đã có sẵn:
[`notebooks/kaggle_train.ipynb`](../notebooks/kaggle_train.ipynb).

## Bước 1 — Đưa code lên Kaggle (chọn 1 trong 2)

### Cách A: qua GitHub (khuyến nghị — notebook đã điền sẵn)
Repo đã ở GitHub: `https://github.com/dangquoc123/toanquoc-ocr.git`, và
`REPO_URL` trong notebook **đã trỏ sẵn vào đó** — không cần cấu hình gì thêm.
Mỗi lần sửa code local, chỉ cần `git push` rồi bấm Run All lại trên Kaggle.

### Cách B: upload trực tiếp làm Kaggle Dataset (không cần GitHub)
```bash
make kaggle-zip          # tạo dist/vnocr-kaggle.zip
```
1. Vào **kaggle.com → Datasets → New Dataset**.
2. Kéo thả `dist/vnocr-kaggle.zip`, đặt tên ví dụ `vnocr-repo`, bấm **Create**.
3. Trong notebook: **Add Input** → chọn dataset `vnocr-repo` của bạn.
4. Để `REPO_URL = ''` — notebook tự tìm repo trong `/kaggle/input/`.

## Bước 2 — Tạo notebook

1. **kaggle.com → Code → New Notebook**.
2. **File → Import Notebook** → chọn `notebooks/kaggle_train.ipynb` từ máy.

## Bước 3 — Bật GPU + Internet (bắt buộc)

Panel bên phải → **Settings**:
- **Accelerator: GPU T4 x2** (hoặc P100).
- **Internet: On** (cần cho `git clone`/tải font; cách B có thể để Off nếu bỏ cell tải Noto).

## Bước 4 — Chạy

- Sửa cell **Cấu hình** nếu muốn (`N_SYNTH`, `EPOCHS`, corpus lớn…).
- **Run All**. Thứ tự: lấy code → font → đo entropy → **smoke test** (phải in
  `SMOKE TEST PASSED`) → sinh ảnh → train (mỗi epoch in CER/WER + **p_B/p_T**) →
  thử đọc 1 ảnh.
- Chạy nền không cần mở trình duyệt: **Save Version → Save & Run All (Commit)**.
  Kết quả nằm ở tab **Output** của phiên bản đã commit.

## Bước 5 — Tải kết quả về máy

Tab **Output** → tải:
- `recognizer.pt` — model đã train
- `vi.count.pkl` — language model

Dùng local:
```bash
python3 scripts/infer.py page.jpg --ckpt recognizer.pt \
    --syllables data/charset/syllables.txt --lm vi.count.pkl --format json
```

## Lên chất lượng thật (sau khi pipeline chạy thông)

| Việc | Cách |
|---|---|
| Corpus lớn (§6.3) | Upload Wikipedia+báo làm Dataset, trỏ `CORPUS_PATH` vào nó |
| Nhiều ảnh hơn (§6.2) | `N_SYNTH = 100000+`, `EPOCHS = 30+` |
| Data thật (§6.1) | Upload VinText… làm Dataset, nối manifest vào train.txt |
| Ablation §3.2 | Chạy 2 phiên: có/không `--no-interaction`, so **p_T** |
| Tiết kiệm quota | Bắt đầu `N_SYNTH` nhỏ để thông pipeline rồi mới scale |

## Sự cố thường gặp

| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `assert torch.cuda.is_available()` | Chưa bật GPU ở Settings → Accelerator |
| `Không thấy repo` | Quên Add Input dataset, hoặc `REPO_URL` sai |
| `git clone` treo | Internet: Off → bật On, hoặc dùng cách B |
| Hết 12h phiên | Giảm `N_SYNTH`/`EPOCHS`; checkpoint đã ở `/kaggle/working` |
| CUDA OOM | Giảm `BATCH` (128 → 64 → 32) |
