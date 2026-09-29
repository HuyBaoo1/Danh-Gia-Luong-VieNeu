# VieNeu-TTS v3 Turbo Normalization Evaluation

## Environment

- Run ID: `76977cf88ec5`
- Python: `3.11.9`
- OS: `Windows-10-10.0.26200-SP0`
- CPU: `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel`
- RAM: `7.7 GB`
- GPU: `NVIDIA GeForce MX330, 2048 MiB, 511.69`
- Backend: `onnx` / `int8` / `cpu`
- `vieneu`: `3.2.4`
- `sea-g2p`: `0.8.4`
- `onnxruntime`: `1.28.0`

## Pipeline verified

Installed source confirms: paragraph normalization with `punc_norm=True`, then normalized-text chunking, then `phonemize_text_with_emotions`, ONNX generation, boundary-aware joining, and segment watermarking.

The v3 path normalizes each chunk again inside the SEA pipeline. The dump therefore includes the first normalized chunk, second-pass text, actual phonemes, and a G2P-only control.

## Dataset

- `danhgia_tts.txt`: 17 segments, 150 chunks, SHA-256 `40F0AEAE73021A4C6447FDD39D345638AFA6AB0FE55D4F596E69E3D5E8B5F163`
- `danhgia_tts_02.txt`: 16 segments, 106 chunks, SHA-256 `BBE614E05AD0A815D688C3F4B44F691E32398694BF574A75E1012775F7936351`
- Total segments: 33
- Total chunks: 256
- Special-case candidates: 1877

## Results

- Generated chunks: 256
- Failed chunks: 0
- NORMALIZE_ERROR: 6
- normalizer: 6
- g2p: 0
- MODEL_ERROR: 0
- UNCERTAIN: 250
- NO_ERROR: 0

Generation-only attribution remains `UNCERTAIN` until a human records the observed pronunciation. ASR is not used as ground truth.

## Important cases

### danhgia_tts-007-c003

- Raw: Các nhà nghiên cứu tại Anthropic của Mỹ cảnh báo rằng các mô hình AI ngày càng mạnh có thể thoát khỏi kiểm soát của con người và thậm chí dẫn tới tuyệt chủng, cảnh báo này đã thu hút sự chú ý ở Trung Quốc. Bộ trưởng An ninh Nhà nước Trung Quốc Trần Nhất Tân...
- Normalized: china cyberspace rằng các mô hình tiên tiến của mỹ như mythos của anthropic và gờ phê tê gạch nối năm chấm năm gạch nối cyber của open <en>a i</en> có thể đe dọa cơ sở hạ tầng thông.
- Phoneme: tʃˈaɪnə sˈaɪbɚspˌeɪs ɹˈa2ŋ kˌaːɜc mˈo hˈi2ɲ t̪ˈiɛn t̪ˈiɛɜn kˌuə4 mˈi5 ɲˌy mˈɪθoʊz kˌuə4 ænθɹˈɑːpɪk vˌaː2 ɣˈəː2 fˈe t̪ˈe ɣˈe-6c nˈoɜj nˈam tʃˈəɜm nˈam ɣˈe-6c nˈoɜj sˈaɪbɚ kˌuə4 ˈoʊpən ˈeɪ ˈaɪ kˈɔɜ tˈe4 ɗˈɛ zˈoaː6 kˈəː sˈəː4 hˈaː6 t̪ˈə2ŋ tˈoŋ.
- Expected: gi pi ti năm chấm năm cyber
- Observed: not audio-reviewed
- Diagnosis: `NORMALIZE_ERROR / normalizer`
- Proof: Hyphenated GPT model name is expanded as Vietnamese letters and literal hyphens while GPT-6 in the second file is normalized to English letter names. Proof: RAW GPT-5.5-Cyber -> NORMALIZED gờ phê tê gạch nối năm chấm năm gạch nối cyber; same baseline maps GPT-6 to <en>g p t</en> sáu cyber.

### danhgia_tts_02-004-c002

- Raw: Thủ tướng Ý Giorgia Meloni hôm 24 tháng 9 tại Rome ban hành sắc lệnh cấm burqa và niqab trong trường học và áp đặt giới hạn 30% học sinh không có khả năng tiếng Ý đủ tốt trong mỗi lớp. Theo dự thảo nghị định, giới hạn 30% sẽ áp dụng từ năm học 2027-28 và dự...
- Normalized: theo dự thảo nghị định, giới hạn ba mươi phần trăm sẽ áp dụng từ năm học hai nghìn không trăm hai mươi bảy hai mươi tám và dựa trên năng lực ngôn ngữ chứ không phải quốc tịch, với một số phiên bản nêu áp dụng cho học sinh không phải công dân ý,
- Phoneme: tˈɛw zˈy6 tˈaː4w ŋˈi6 ɗˈi6ɲ, zˈəːɜj hˈaː6n bˈaː mˈyəj fˈə2n tʃˈam sˌɛ5 ˈaːɜp zˈu6ŋ t̪ˌy2 nˈam hˈɔ6k hˈaːj ŋˈi2n xˌoŋ tʃˈam hˈaːj mˈyəj bˈa4j hˈaːj mˈyəj t̪ˈaːɜm vˌaː2 zˈyə6 tʃˈen nˈaŋ lˈy6c ŋˈon ŋˈy5 tʃˈyɜ xˌoŋ fˌaː4j kˈuəɜc t̪ˈi6c, vˌəːɜj mˈo6t̪ sˈoɜ fˈiɛn bˈaː4n nˈe1w ˈaːɜp zˈu6ŋ tʃˌɔ hˈɔ6k sˈiɲ xˌoŋ fˌaː4j kˈoŋ zˈən ˈiɜ.
- Expected: năm học hai nghìn không trăm hai mươi bảy đến hai nghìn không trăm hai mươi tám
- Observed: not audio-reviewed
- Diagnosis: `NORMALIZE_ERROR / normalizer`
- Proof: The abbreviated school-year range loses its range relation. Proof: RAW 2027-28 -> NORMALIZED hai nghìn không trăm hai mươi bảy hai mươi tám; the word đến is absent.

### danhgia_tts_02-008-c001

- Raw: Tổng Bí thư, Chủ tịch nước Tô Lâm của Việt Nam thăm cấp Nhà nước tới Canada từ ngày 24-25/9/2026 theo lời mời của Toàn quyền Canada Louise Arbour. Trong khuôn khổ chuyến thăm, ông Tô Lâm đã hội đàm với Thủ tướng Canada Mark Carney tại Ottawa và hai bên nhất...
- Normalized: tổng bí thư, chủ tịch nước tô lâm của việt nam thăm cấp nhà nước tới canada từ ngày hai mươi bốn ngày hai mươi lăm tháng chín năm hai nghìn không trăm hai mươi sáu theo lời mời của toàn quyền canada louise arbour.
- Phoneme: t̪ˈo4ŋ bˈiɜ tˈy, tʃˈu4 t̪ˈi6c nˈyəɜc t̪ˈo lˈəm kˌuə4 vˈiɛ6t̪ nˈaːm tˈam kˈəɜp ɲˈaː2 nˈyəɜc t̪ˌəːɜj kˈænədə t̪ˌy2 ŋˈa2j hˈaːj mˈyəj bˈoɜn ŋˈa2j hˈaːj mˈyəj lˈam tˈaːɜŋ tʃˈiɜn nˈam hˈaːj ŋˈi2n xˌoŋ tʃˈam hˈaːj mˈyəj sˈaɜw tˈɛw lˈəː2j mˈəː2j kˌuə4 t̪wˈaː2n kwˈiɛ2n kˈænədə luːwˈiːz ˈɑːɹbɚ.
- Expected: từ ngày hai mươi bốn đến ngày hai mươi lăm tháng chín năm hai nghìn không trăm hai mươi sáu
- Observed: not audio-reviewed
- Diagnosis: `NORMALIZE_ERROR / normalizer`
- Proof: The compact date range loses its range relation. Proof: RAW từ ngày 24-25/9/2026 -> NORMALIZED từ ngày hai mươi bốn ngày hai mươi lăm tháng chín năm hai nghìn không trăm hai mươi sáu; the word đến is absent.

### danhgia_tts_02-011-c003

- Raw: Xiaomi đã giới thiệu dòng điện thoại Pro 18 tại Snapdragon Summit ở Hawaii, trong đó 18 Pro Max chạy phiên bản cao cấp hơn của chip Qualcomm Snapdragon 8 Elite Gen 6. Điểm nổi bật là màn hình tập trung vào quyền riêng tư, tương tự công nghệ của Samsung, cho...
- Normalized: màn hình chính là lờ tê phê ô a mờ ô lờ e đê hai ca, biến thiên một một trăm hai mươi héc, độ sáng đỉnh bốn nghìn nits, mười tám pro có màn sáu phẩy bốn inch, mười tám pro max sáu phẩy chín inch, kèm màn hình phụ phía sau,
- Phoneme: mˈaː2n hˈi2ɲ tʃˈiɜɲ lˌaː2 lˈəː2 t̪ˈe fˈe ˈo ˈaː mˈəː2 ˈo lˈəː2 ˈɛ ɗˈe hˈaːj kˈaː, bˈiɛɜn tˈiɛn mˈo6t̪ mˈo6t̪ tʃˈam hˈaːj mˈyəj hˈɛɜc, ɗˈo6 sˈaːɜŋ ɗˈi4ɲ bˈoɜn ŋˈi2n nˈɪts, mˈyə2j t̪ˈaːɜm pɹˈoʊ kˈɔɜ mˈaː2n sˈaɜw fˈəɪ4 bˈoɜn ˈɪntʃ, mˈyə2j t̪ˈaːɜm pɹˈoʊ mˈæks sˈaɜw fˈəɪ4 tʃˈiɜn ˈɪntʃ, kˈɛ2m mˈaː2n hˈi2ɲ fˈu6 fˈiəɜ sˈaw.
- Expected: biến thiên một đến một trăm hai mươi héc
- Observed: not audio-reviewed
- Diagnosis: `NORMALIZE_ERROR / normalizer`
- Proof: The refresh-rate range loses its range relation. Proof: RAW 1-120Hz -> NORMALIZED một một trăm hai mươi héc; the word đến is absent.

### danhgia_tts_02-013-c002

- Raw: Thủ tướng Campuchia Hun Manet đã quyết định triển khai chương trình trợ cấp tiền mặt của chính phủ nhằm hỗ trợ các gia đình nghèo, dễ bị tổn thương và dễ gặp rủi ro chịu ảnh hưởng từ áp lực tăng giá dầu và khí đốt. Chương trình sẽ bắt đầu phát tiền vào ngày...
- Normalized: chương trình sẽ bắt đầu phát tiền vào ngày hai mươi lăm tháng chín năm hai nghìn không trăm hai mươi sáu và thực hiện bốn đợt trong năm hai nghìn không trăm hai mươi sáu, với các ngày cụ thể là hai mươi lăm trên chín, mười sáu trên mười,
- Phoneme: tʃˈyəŋ tʃˈi2ɲ sˌɛ5 bˈaɜt̪ ɗˈə2w fˈaːɜt̪ t̪ˈiɛ2n vˈaː2w ŋˈa2j hˈaːj mˈyəj lˈam tˈaːɜŋ tʃˈiɜn nˈam hˈaːj ŋˈi2n xˌoŋ tʃˈam hˈaːj mˈyəj sˈaɜw vˌaː2 tˈy6c hˈiɛ6n bˈoɜn ɗˈəː6t̪ tʃˈɔŋ nˈam hˈaːj ŋˈi2n xˌoŋ tʃˈam hˈaːj mˈyəj sˈaɜw, vˌəːɜj kˌaːɜc ŋˈa2j kˈu6 tˈe4 lˌaː2 hˈaːj mˈyəj lˈam tʃˈen tʃˈiɜn, mˈyə2j sˈaɜw tʃˈen mˈyə2j.
- Expected: các ngày cụ thể là ngày hai mươi lăm tháng chín và ngày mười sáu tháng mười
- Observed: not audio-reviewed
- Diagnosis: `NORMALIZE_ERROR / normalizer`
- Proof: Slash dates are read as fractions despite explicit date context. Proof: RAW các ngày cụ thể là 25/9 16/10 -> NORMALIZED hai mươi lăm trên chín mười sáu trên mười.

### danhgia_tts_02-013-c003

- Raw: Thủ tướng Campuchia Hun Manet đã quyết định triển khai chương trình trợ cấp tiền mặt của chính phủ nhằm hỗ trợ các gia đình nghèo, dễ bị tổn thương và dễ gặp rủi ro chịu ảnh hưởng từ áp lực tăng giá dầu và khí đốt. Chương trình sẽ bắt đầu phát tiền vào ngày...
- Normalized: mười sáu trên mười một và mười sáu trên mười hai.
- Phoneme: mˈyə2j sˈaɜw tʃˈen mˈyə2j mˈo6t̪ vˌaː2 mˈyə2j sˈaɜw tʃˈen mˈyə2j hˈaːj.
- Expected: ngày mười sáu tháng mười một và ngày mười sáu tháng mười hai
- Observed: not audio-reviewed
- Diagnosis: `NORMALIZE_ERROR / normalizer`
- Proof: Slash dates are read as fractions despite explicit date context. Proof: RAW 16/11 và 16/12 -> NORMALIZED mười sáu trên mười một và mười sáu trên mười hai.

## Conclusion

Confirmed preprocessing errors are reported separately from generation errors. Any generated audio that has not been listened to remains `UNCERTAIN`; this is deliberate evidence discipline, not a missing sample.
