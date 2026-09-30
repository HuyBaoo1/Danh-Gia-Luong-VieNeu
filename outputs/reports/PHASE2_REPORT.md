# VieNeu-TTS v3 Turbo - Phase 2 Human Pronunciation Review

## Baseline

- Project status: `COMPLETE`
- Run ID: `76977cf88ec5`
- Model: `pnnbao-ump/VieNeu-TTS-v3-Turbo`
- Model revision: `067cf9d6abaf3ddb4de78135467aaa81502f0a6d`
- VieNeu version: `3.2.4`
- sea-g2p version: `0.8.4`
- Backend: `onnx / int8 / cpu`
- Voice: `Minh Đức`
- Total segments: 33
- Total chunks: 256

Normalization findings apply specifically to the pinned benchmark environment above.

## Review Coverage

- Segments reviewed: 33/33
- Chunks reviewed: 256/256
- Pending: 0
- Needs recheck: 0
- Review coverage: 100.00%
- Priority subset: 0 chunks across 0 segments
- Priority subset reviewed: 0/0

## Final Error Counts

These are current classifications. The six Phase 1 normalization findings are carried forward, but remain pending human listening unless listed as reviewed.

- NO_ERROR = 250
- NORMALIZE_ERROR = 6
- G2P_ERROR = 0
- MODEL_ERROR = 0
- UNCERTAIN = 0
- MODEL_ERROR_RATE = 0.00%

## Confirmed Model Errors

No MODEL_ERROR has been confirmed so far.

## Normalization Errors Confirmed During Listening

- `danhgia_tts-007-c003`: expected 'gi pi ti năm chấm năm cyber', observed 'gi phi ti gạch nối năm chấm năm gạch nối cyber' (`normalizer`, HIGH).
- `danhgia_tts_02-004-c002`: expected 'năm học hai nghìn không trăm hai mươi bảy đến hai nghìn không trăm hai mươi tám', observed 'năm học hai nghìn không trăm hai mươi bảy hai mươi tám' (`normalizer`, HIGH).
- `danhgia_tts_02-008-c001`: expected 'từ ngày hai mươi bốn đến ngày hai mươi lăm tháng chín năm hai nghìn không trăm hai mươi sáu', observed 'từ ngày hai mươi bốn ngày hai mươi lăm tháng chín năm hai nghìn không trăm hai mươi sáu' (`normalizer`, HIGH).
- `danhgia_tts_02-011-c003`: expected 'biến thiên một đến một trăm hai mươi héc', observed 'biến thiên một một trăm hai mươi héc' (`normalizer`, HIGH).
- `danhgia_tts_02-013-c002`: expected 'các ngày cụ thể là ngày hai mươi lăm tháng chín và ngày mười sáu tháng mười', observed 'các ngày cụ thể là ngày hai mươi lăm trên chín, mươi sáu trên mười' (`normalizer`, HIGH).
- `danhgia_tts_02-013-c003`: expected 'ngày mười sáu tháng mười một và ngày mười sáu tháng mười hai', observed 'mười sáu trên mười một và mười sáu trên mươi hai' (`normalizer`, HIGH).

## Phase 1 Findings Awaiting Listening


## Uncertain Cases

0 chunks remain `UNCERTAIN`; use `segment_review_index.csv` for segment-first listening and `human_review.csv` for chunk-level attribution.

ASR triage was not used. Human perceptual judgment remains the ground truth.

## Conclusion

PROJECT_STATUS = COMPLETE
