# VieNeu-TTS v3 Turbo Evaluation

## Mục tiêu

Xác định lỗi phát âm đến từ **text normalization/G2P** hay từ **TTS generation model** theo chuỗi:

```text
Raw text -> Normalized text -> Phoneme -> Generated audio
```

## Baseline

- Model: `pnnbao-ump/VieNeu-TTS-v3-Turbo`
- Model revision: `067cf9d6abaf3ddb4de78135467aaa81502f0a6d`
- VieNeu: `3.2.4`
- sea-g2p: `0.8.4`
- Backend: ONNX int8 / CPU
- Voice: Minh Đức
- Run ID: `76977cf88ec5`
- Dataset: 33 segments, 256 chunks
- Audio: 289 WAV, 48 kHz mono

## Kết quả

| Phân loại | Số lượng |
|---|---:|
| NO_ERROR | 250 |
| NORMALIZE_ERROR | 6 |
| G2P_ERROR | 0 |
| MODEL_ERROR | 0 |
| UNCERTAIN | 0 |

- Human review: `256/256` chunks (`100%`)
- Model error rate trong tập review: `0.00%`

## Normalization Errors

Sáu lỗi được xác nhận liên quan đến:

- `GPT-5.5-Cyber`
- khoảng năm `2027-28`
- khoảng ngày `24-25/9/2026`
- khoảng tần số `1-120Hz`
- ngày dạng slash: `25/9`, `16/10`, `16/11`, `16/12`

## Kết luận

Xác nhận **6 lỗi normalization**. Không có `G2P_ERROR` hoặc `MODEL_ERROR` được xác nhận trong tập đánh giá.

## Kết quả chi tiết

- [Phase 2 report](outputs/reports/PHASE2_REPORT.md)
- [Human review](outputs/reports/human_review.csv)
- [Phase 1 report](outputs/reports/REPORT.md)
- [Error analysis](outputs/reports/error_analysis.csv)
- Audio được lưu trong `outputs/audio/` và `outputs/chunks/` bằng Git LFS.
