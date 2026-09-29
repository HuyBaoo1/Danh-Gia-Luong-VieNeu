from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sea_g2p import Normalizer

from src.human_review import merge_review_row, validate_review_row
from src.inspect_normalization import parse_segments, scan_candidates
from src.schemas import ErrorStage, ErrorType, Segment, validate_attribution
from src.utils import stable_seed


class InputParsingTests(unittest.TestCase):
    def test_parses_labeled_segments_without_collapsing_internal_newlines(self) -> None:
        text = "Văn bản 1:\r\nDòng một.\r\nDòng hai.\r\n\r\nVăn bản 2:\r\nĐoạn hai.\r\n"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "input.txt"
            path.write_bytes(text.encode("utf-8"))
            segments, metadata = parse_segments(path)

        self.assertEqual(2, len(segments))
        self.assertEqual("Dòng một.\r\nDòng hai.", segments[0].raw_text)
        self.assertEqual("Đoạn hai.", segments[1].raw_text)
        self.assertEqual("UTF-8", metadata["encoding"])

    def test_candidate_scan_flags_date_range_and_percentage(self) -> None:
        segment = Segment(
            source_file="fixture.txt",
            segment_id="fixture-001",
            source_label="Văn bản 1",
            raw_text="Áp dụng 14/09-15/09 với mức 10-15%.",
            source_sha256="fixture",
        )
        categories = {row.category for row in scan_candidates(segment)}
        self.assertIn("date_range", categories)
        self.assertIn("percentage_range", categories)


class AttributionTests(unittest.TestCase):
    def test_accepts_only_supported_type_stage_pairs(self) -> None:
        validate_attribution(
            ErrorType.NORMALIZE_ERROR.value, ErrorStage.NORMALIZER.value
        )
        validate_attribution(ErrorType.MODEL_ERROR.value, ErrorStage.GENERATION.value)
        with self.assertRaises(ValueError):
            validate_attribution(
                ErrorType.MODEL_ERROR.value, ErrorStage.NORMALIZER.value
            )

    def test_chunk_seed_is_stable_and_chunk_specific(self) -> None:
        first = stable_seed(123, "sample-c001")
        self.assertEqual(first, stable_seed(123, "sample-c001"))
        self.assertNotEqual(first, stable_seed(123, "sample-c002"))

    def test_phase2_resume_preserves_human_fields(self) -> None:
        base = {
            "chunk_id": "sample-c001",
            "review_status": "PENDING",
            "error_type": "UNCERTAIN",
            "error_stage": "uncertain",
            "severity": "",
            "reviewer_note": "",
        }
        existing = {
            "review_status": "REVIEWED",
            "error_type": "NO_ERROR",
            "error_stage": "none",
            "severity": "LOW",
            "pronunciation_correct": "true",
            "reviewer_note": "Listened to the baseline WAV.",
        }
        row = merge_review_row(base, None, existing)
        self.assertEqual("REVIEWED", row["review_status"])
        self.assertEqual("NO_ERROR", row["error_type"])
        self.assertEqual("Listened to the baseline WAV.", row["reviewer_note"])
        validate_review_row(row)

    def test_model_error_requires_two_reruns(self) -> None:
        row = {
            "chunk_id": "sample-c001",
            "review_status": "REVIEWED",
            "error_type": "MODEL_ERROR",
            "error_stage": "generation",
            "severity": "HIGH",
            "pronunciation_correct": "false",
            "expected_reading": "expected",
            "observed_reading": "observed",
            "rerun_count": "1",
            "reproducible": "true",
            "rerun_result": "same error",
        }
        with self.assertRaisesRegex(ValueError, "two diagnostic reruns"):
            validate_review_row(row)

    def test_uncertain_review_requires_recheck_status(self) -> None:
        row = {
            "chunk_id": "sample-c001",
            "review_status": "REVIEWED",
            "error_type": "UNCERTAIN",
            "error_stage": "uncertain",
            "severity": "",
            "pronunciation_correct": "",
        }
        with self.assertRaisesRegex(ValueError, "NEEDS_RECHECK"):
            validate_review_row(row)


class BaselineBehaviorProbeTests(unittest.TestCase):
    def test_installed_normalizer_date_range_snapshot(self) -> None:
        # This is a captured baseline symptom, not the desired correction. It makes
        # a sea-g2p behavior change immediately visible in the diagnostic loop.
        normalized = Normalizer("vi").normalize("6/8/1976-6/8", punc_norm=True)
        self.assertEqual(
            "ngày sáu tháng tám năm một nghìn chín trăm bảy mươi sáu sáu trên tám.",
            normalized,
        )


if __name__ == "__main__":
    unittest.main()

