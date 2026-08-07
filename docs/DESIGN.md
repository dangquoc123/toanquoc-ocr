# Thiết kế hệ thống OCR tiếng Việt — nhẹ, chính xác, không phụ thuộc LLM

> Đây là **spec of record** của dự án. Mã nguồn tham chiếu tới các mục (§) trong
> tài liệu này ở khắp docstring. Nguyên tắc xuyên suốt: **thắng trong miền hẹp
> (văn bản in tiếng Việt + bảng hành chính) bằng chuyên biệt hoá, không bằng quy
> mô.** Mọi thành phần chạy trên CPU, tổng mạng nơ-ron < 30MB, và không có LLM ở
> bất kỳ đâu trong đường suy luận.

Con số định cỡ tham chiếu (đo độc lập trên tiếng Việt, tháng 5/2026): recognizer
chuyên tiếng Việt như VietOCR đạt CER ~1,41% trên chữ in — vượt cả VLM 72B
(CER ~10,7%). PaddleOCR mặc định đạt 24,70% CER chỉ vì dùng recognizer Latin rớt
dấu. Khoảng cách 20+ điểm phần trăm đó nằm hoàn toàn ở việc huấn luyện recognizer
riêng cho dấu — đó là nơi dự án này giành điểm.

---

## §0 — Vì sao không cần LLM (nền tảng chi phí)

Lỗi OCR tiếng Việt bị **thanh điệu chi phối**: chữ nền (a-z, số) đọc gần đúng,
nhưng dấu sắc/huyền/hỏi/ngã/nặng sai ở tỉ lệ cao. Ký hiệu: `p_B` = tỉ lệ sai
khung âm-tự ≈ 0,5%; `p_T` = tỉ lệ sai thanh ≈ 4%. Thanh điệu áp đảo.

Cái gì phục hồi được thanh điệu? Từ điển đơn âm tiết thì **không** — thay thanh
một âm tiết hợp lệ hầu như luôn ra một âm tiết hợp lệ khác (ma/má/mà/mã/mạ/mả đều
thật), khoảng cách mã bằng 1. Chỉ **ngữ cảnh** mới phục hồi được.

Điểm mấu chốt: ngữ cảnh cần thiết là **bigram/trigram thống kê**, KHÔNG phải LLM.
Bất đẳng thức Fano: khi `H(thanh | ngữ cảnh)` nhỏ (0,05–0,15 bit cho tiếng Việt),
bộ giải mã MAP dùng bigram nằm trong ~1,35 lần giới hạn lý thuyết. Một LLM 7B cũng
không vượt sàn Fano đó đáng kể. **LLM ở đây là phí tiền mua thứ n-gram đã gần chạm
trần lý thuyết.**

Hệ quả: tách "hiểu ngôn ngữ" khỏi mạng nhận dạng, đặt vào bộ hậu xử lý thống kê
(KenLM n-gram + trie âm tiết + FST). Nặng vài chục MB, CPU, dưới mili-giây, huấn
luyện bằng đếm tần suất trên văn bản thuần — không GPU, không gán nhãn, tự chủ
hoàn toàn.

---

## §1 — Tiền xử lý ảnh

### §1.1 Đính chính: KHÔNG nhị phân hoá cứng cho recognizer deep learning
Nhị phân hoá (Otsu, Sauvola) chuẩn cho Tesseract cổ điển nhưng **có hại** cho
SVTR/transformer: dấu thanh là thành phần tần số cao với biên chuyển màu mờ; ép về
0/1 xoá dải xám ở biên — chính thông tin phân biệt hỏi (một móc) với ngã (nét
lượn). Recognizer học sâu ăn **grayscale/màu**. Nhị phân hoá CHỈ dùng cho phát
hiện đường kẻ bảng (§4).

### §1.2–1.6 Chuỗi tiền xử lý
1. **Chuẩn hoá DPI** — mục tiêu 300–400 DPI; dưới 200 DPI dấu thanh rơi dưới ngưỡng
   phân giải; siêu phân giải nhẹ nếu cần; trên 600 hạ mẫu.
2. **Khử nghiêng (deskew)** — Hough hoặc phương sai phép chiếu, quét -15°..+15°.
3. **Khử cong (dewarp)** — chỉ khi cần (ảnh điện thoại, sách mở).
4. **Chuẩn hoá tương phản** — CLAHE (cục bộ), giữ grayscale.
5. **Khử nhiễu nhẹ** — bilateral (giữ biên), không Gaussian.
6. **Chuẩn hoá Unicode nhãn (NFC)** — bắt buộc; "ế" dựng sẵn vs tổ hợp; chuẩn hoá
   cả kiểu đặt dấu cũ/mới (hoà → hòa).

---

## §2 — Kiến trúc tổng thể

```
Ảnh → [Tiền xử lý] → [Phân tích bố cục] → ├─ nhánh chữ ─→ [Phát hiện] → [Nhận dạng] → [Hậu xử lý n-gram]
                                          └─ nhánh bảng ─→ [Cấu trúc bảng] → [Gán ô]
                                                                    ↓
                                                          [Hợp nhất → HTML/JSON/Markdown]
```

Modular chứ không end-to-end VLM: pipeline tách khối **không sinh ngôn ngữ nên
không thể ảo giác**, cho bounding box chính xác, mỗi khối tối ưu/gỡ lỗi riêng.

| Khối | Thuật toán | Tham số | Dung lượng (FP32→INT8) |
|---|---|---|---|
| Phân tích bố cục | PP-DocLayout-lite / YOLO-doc | ~5M | 5MB → 2MB |
| Phát hiện chữ | DBNet + LCNetV4 + RepLKFPN | 3–5M | 4MB → 1,5MB |
| Nhận dạng chữ | SVTR + đầu tách dấu + CTC (GTC-NRTR) | 8–12M | 10MB → 3MB |
| Cấu trúc bảng | Hình thái học (0 tham số) + SLANet | ~1M | 3MB → 1MB |
| Hậu xử lý | KenLM bigram + trie + FST | 0 (bảng đếm) | ~20–40MB |

Tổng mạng nơ-ron: < 30MB FP32, ~8–10MB sau INT8.

---

## §3 — Khối nhận dạng (trái tim)

### §3.1 GTC (Guided Training of CTC)
Nhánh thầy GTC-NRTR (seq2seq attention, có `p(y_u|y_<u)`) chưng cất sang nhánh trò
SVTR+CTC. `L = λ₁·L_CTC + λ₂·L_KD`. **Suy luận vứt nhánh thầy** — độ chính xác gần
attention với tốc độ CTC; encoder dùng chung được "định hình" tốt hơn.

### §3.2 Đòn bẩy 1 — Đầu tách thành phần dấu (phân rã logit)
Thay softmax phẳng ~230 lớp (khiến ữ, ặ, ỡ chết đói dữ liệu), phân rã tầng logit:
```
s(b,m,t | h) = u_b(h) + v_m(h) + w_t(h) + β(b,m,t)
p(b,m,t | h) = M(b,m,t) · exp(s) / Σ M · exp(s)
```
- b = chữ nền (~70–100 lớp), m = dấu phụ nguyên âm (5), t = thanh (6)
- M = mặt nạ tổ hợp hợp lệ ∈ {0,1}
- **Một CTC duy nhất, một căn chỉnh duy nhất** — chỉ tầng chiếu phân rã.

Cỡ mẫu hiệu dụng: vector w_ngã nhận gradient từ MỌI ký tự mang thanh ngã (~10⁻¹)
thay vì chỉ mẫu chứa ữ (~10⁻³–10⁻⁴) → bằng chứng huấn luyện tăng 2–3 bậc.
**Trung thực:** giả định cộng tính không hoàn hảo (β là hiệu chỉnh); **phải
ablation** phẳng vs phân rã — đây là phần duy nhất chưa được chứng minh thực
nghiệm, làm sau cùng.

### §3.3 Đòn bẩy 2 — Chiều cao 48px và stride bất đối xứng
48px chứ không 32px kế thừa Latin: dấu thanh ~0,10–0,13 em; ở 32px ≈ 2,4–3,1px
(dưới Nyquist để phân biệt hỏi/ngã), ở 48px ≈ 3,6–4,7px (vừa đủ). **Stride bất
đối xứng quan trọng hơn:** dùng (1,2) ở tầng đầu — hạ chiều rộng trước, giữ chiều
cao lâu — chỉ hạ chiều cao ở tầng sau. Chi phí ~0, lợi ích rõ.

### §3.4 Bộ ký tự và từ điển
Dictionary tiếng Việt đầy đủ để xây mặt nạ M và tập âm tiết hợp lệ cho hậu xử lý.

---

## §4 — Khối bảng biểu (lai theo bất biến)

### §4.1 Bảng kẻ viền đầy đủ → hình thái học (sai số 0 toán học)
Nhị phân hoá → mở hình thái nhân ngang/dọc → giao điểm H∧V → nút → **mỗi ô = một
mặt của đồ thị phẳng**. Bất biến hình học chính xác, O(số điểm ảnh), 0 tham số.

### §4.2 Bảng không viền / viền đứt → SLANet
~1M tham số, CPU; chỉ kém UniTable Large 0,41% S-TEDS. Cấu trúc bảng **độc lập
ngôn ngữ** — tiền huấn luyện PubTabNet/SynthTabNet, tinh chỉnh vài nghìn bảng Việt.

### §4.3 Gán nội dung vào ô
Chạy recognizer trên từng vùng ô, ghép theo toạ độ → HTML/JSON giữ quan hệ hàng-cột.

---

## §5 — Khối hậu xử lý không-LLM (điểm tự chủ)

### §5.1 Giải mã ràng buộc từ điển trên lattice CTC
KHÔNG sửa sau khi lấy 1-best. Đưa trie âm tiết vào **beam search tiền tố trên
lattice CTC**, ràng buộc FST từ vựng. Tiếng Việt chỉ ~7.000 âm tiết hợp lệ → âm
tiết ngoài tập gần như chắc chắn là lỗi khung (ưu thế cấu trúc tiếng Trung/Anh
không có).

### §5.2 Mô hình kênh nhiễu cho lỗi khung
`ŷ = argmax p(o|y)·p(y)`, `p(o|y)` = khoảng cách sửa có trọng số
`w(a→b) = -log p(quan sát b | thật a)`, **ước lượng từ ma trận nhầm lẫn thực
nghiệm** (không chỉnh tay). Thiên về nhầm lẫn thị giác: hỏi↔ngã, ơ↔o, ư↔u, ê↔e.

### §5.3 Bigram/trigram phục hồi thanh điệu (thay LLM)
`t̂ = argmax_t ℓ(t)·π(t|ngữ cảnh)`, ℓ = bằng chứng thị giác, π = tiên nghiệm KenLM.
Ví dụ lật lỗi: ℓ(hỏi)=0,4, ℓ(ngã)=0,6 (thị giác sai) × π(hỏi)=0,85, π(ngã)=0,15
→ hỏi 0,34 vs ngã 0,09 → chọn **hỏi (đúng)**. ℓ và π độc lập → sai số còn lại ≈
tích ε_thị-giác · ε_ngữ-cảnh. Bigram đã đưa ε_ngữ-cảnh gần sàn Fano.

---

## §6 — Dữ liệu (nơi quyết định thắng thua)

- **§6.1 Nguồn có sẵn:** VinText/Vintext (2.000 ảnh), 5CD-AI/Viet-Handwriting-OCR
  (23.403), HANDS-VNOnDB, Viet-OCR-VQA (137K+).
- **§6.2 Tổng hợp (phần lớn nhất):** SynthTIGER/TRDG + ngữ liệu Việt. **Cảnh báo
  font:** nhiều font Latin dựng dấu Việt SAI — kiểm thủ công ữ, ặ, ỡ, ẫ. Suy giảm
  mô phỏng: JPEG, mờ, in-quét, nghiêng, tương phản thấp, DPI thấp.
- **§6.3 Ngữ liệu n-gram:** văn bản thuần có dấu chuẩn (Wikipedia + báo), không cần ảnh.

---

## §7 — Huấn luyện và tối ưu

- **§7.1 Phần cứng:** suy luận CPU 32GB đủ; huấn luyện cần GPU (Kaggle 30h T4/tuần).
- **§7.2 Bốn kỹ thuật:** (1) chưng cất tri thức (giá trị ở phân phối lớp SAI,
  quý cho bộ ký tự dày dấu); (2) tái tham số hoá cấu trúc (miễn phí độ trễ);
  (3) QAT INT8; (4) tỉa có cấu trúc (sau cùng).
- **§7.3 QAT chứ không PTQ:** biên logit nhỏ (ả/ã, ơ/o vài điểm ảnh); sai số lượng
  tử ~S/2 cùng cỡ biên → đảo argmax. QAT chèn nút giả-lượng-tử để mạng học biên đủ
  rộng. Kiểm: lượng tử hoá rồi đo **riêng độ chính xác thanh điệu**.

---

## §8 — Đánh giá

Ngoài CER/WER, **tách riêng p_B (chữ nền) và p_T (thanh điệu)** — xác nhận
p_T ≫ p_B. Không benchmark công khai nào đo cái này; bạn phải tự dựng. Bảng: TEDS,
TEDS-Struct. Baseline: PP-OCRv6_tiny, VietOCR, Tesseract vie (chuẩn hoá 300 DPI).
Đo entropy để định trần: `H(t)`, `H(t|s)`, `H(t|s,w₋₁)` hiệu chỉnh Miller-Madow.
Nếu mức cuối 0,05–0,15 bit → sàn thanh ~0,5–1%, ngân sách <1,5% đứng vững; cao hơn
→ cần trigram. Phép đo báo TRƯỚC khi huấn luyện.

---

## §9 — Lộ trình triển khai

- **Giai đoạn 1 (2–4 tuần):** Fork PaddleOCR, tinh chỉnh PP-OCRv6_tiny cho tiếng
  Việt bằng dữ liệu tổng hợp. Dựng NGAY bộ đánh giá tách p_B/p_T và chuẩn hoá NFC.
- **Giai đoạn 2 (1–2 tháng):** Thêm ba đòn bẩy + nhánh bảng lai. Đây là chỗ tách
  khỏi mọi fork "PaddleOCR cho tiếng Việt".
- **Giai đoạn 3:** Chưng cất, QAT INT8, xuất ONNX, đóng gói (FastAPI + ONNX Runtime).

Thứ tự kiểm chứng theo chi phí/lợi ích: (1) 48px + stride bất đối xứng — rẻ nhất;
(2) giải mã ràng buộc từ điển — không cần huấn luyện lại; (3) hậu xử lý bigram —
KenLM CPU; (4) phân rã logit dấu — cần huấn luyện lại + ablation, làm sau cùng.

---

## Phụ lục — Điều thành thật phải nói

- Hệ thống **không "đánh bại PP-OCRv6 trên mọi mặt"** — thắng miền hẹp bằng cách
  hy sinh tính tổng quát.
- **Phân rã logit dấu (§3.2) là giả thuyết chưa kiểm** — ba đòn bẩy còn lại vững
  hơn. Phải ablation.
- Lợi thế **phụ thuộc chất lượng dữ liệu tổng hợp và font phủ đủ dấu**.
- **Không có benchmark công khai đo DeepSeek-OCR/Paddle trên tiếng Việt** — mọi so
  sánh phải tự chạy trên tập gán nhãn của bạn.
- Con số ngân sách (p_B≈0,5%, p_T≈4%, sàn Fano ~0,74%) là **ước lượng minh hoạ** —
  thay số đo thật vào công thức để ra ngân sách thật.
