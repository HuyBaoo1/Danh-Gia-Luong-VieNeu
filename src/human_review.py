from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schemas import ErrorStage, ErrorType, validate_attribution
from .utils import load_config, read_csv, read_jsonl, run_id, write_csv


REVIEW_STATUSES = {"PENDING", "REVIEWED", "NEEDS_RECHECK"}
SEVERITIES = {"", "LOW", "MEDIUM", "HIGH"}

HUMAN_REVIEW_FIELDS = [
    "source_file",
    "segment_id",
    "chunk_id",
    "raw_text",
    "normalized_text",
    "phonemes",
    "audio_segment_path",
    "audio_chunk_path",
    "expected_reading",
    "observed_reading",
    "pronunciation_correct",
    "error_type",
    "error_stage",
    "severity",
    "review_status",
    "reviewer_note",
    "priority_subset",
    "subset_rank",
    "subset_score",
    "subset_reason",
    "review_priority",
    "candidate_categories",
    "segment_duration_seconds",
    "chunk_duration_seconds",
    "rerun_count",
    "reproducible",
    "rerun_result",
]

HUMAN_EDITABLE_FIELDS = {
    "expected_reading",
    "observed_reading",
    "pronunciation_correct",
    "error_type",
    "error_stage",
    "severity",
    "review_status",
    "reviewer_note",
    "rerun_count",
    "reproducible",
    "rerun_result",
}

SEGMENT_REVIEW_FIELDS = [
    "source_file",
    "segment_id",
    "audio_segment_path",
    "raw_text",
    "chunk_ids",
    "chunk_count",
    "review_priority",
    "candidate_categories",
    "duration_seconds",
    "priority_subset_chunks",
    "priority_subset_count",
    "priority_subset_best_rank",
    "segment_review_status",
    "suspicious_chunk_ids",
    "reviewer_note",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _priority(categories: set[str]) -> str:
    p1 = {
        "english_or_foreign_word",
        "acronym",
        "number",
        "decimal",
        "hyphen_expression",
        "slash_expression",
    }
    p3 = {
        "date",
        "date_range",
        "time_range",
        "measurement",
        "currency",
        "currency_per_unit",
        "percentage",
        "percentage_range",
    }
    if categories & p1:
        return "P1"
    if "mixed_vietnamese_english" in categories:
        return "P2"
    if categories & p3:
        return "P3"
    return "P4"


def _audio_metadata(
    output_root: Path, current_run_id: str, kind: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    root = output_root / kind / current_run_id
    if not root.exists():
        return result
    key = "chunk_id" if kind == "chunks" else "segment_id"
    for path in root.rglob("*.json"):
        metadata = _load_json(path)
        identifier = metadata.get(key, "")
        if identifier and metadata.get("status") == "ok":
            result[identifier] = metadata
    return result


def _match_text(value: str) -> str:
    value = value.strip().lower().rstrip(".!?,;:")
    return re.sub(r"\s+", " ", value)


def _candidate_categories(
    report_root: Path, chunks: list[dict[str, Any]]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    segment_categories: dict[str, set[str]] = defaultdict(set)
    chunk_categories: dict[str, set[str]] = defaultdict(set)
    chunks_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_segment[chunk["segment_id"]].append(chunk)

    for candidate in read_csv(report_root / "special_cases.csv"):
        segment_id = candidate["segment_id"]
        category = candidate["category"]
        segment_categories[segment_id].add(category)
        needles = {
            _match_text(candidate.get("isolated_normalized", "")),
            _match_text(candidate.get("matched_text", "")),
            _match_text(
                candidate.get("matched_text", "").replace("-", " ").replace("/", " ")
            ),
        }
        needles.discard("")
        for chunk in chunks_by_segment.get(segment_id, []):
            haystack = _match_text(chunk["chunk_normalized_text"])
            if any(needle in haystack for needle in needles):
                chunk_categories[chunk["chunk_id"]].add(category)
    return segment_categories, chunk_categories


def _base_review_row(
    chunk: dict[str, Any],
    segment_audio: dict[str, Any],
    chunk_audio: dict[str, Any],
    categories: set[str],
) -> dict[str, Any]:
    return {
        "source_file": chunk["source_file"],
        "segment_id": chunk["segment_id"],
        "chunk_id": chunk["chunk_id"],
        "raw_text": chunk["raw_text"],
        "normalized_text": chunk["chunk_normalized_text"],
        "phonemes": chunk["phonemes"],
        "audio_segment_path": segment_audio.get("audio_path", ""),
        "audio_chunk_path": chunk_audio.get("audio_path", ""),
        "expected_reading": "",
        "observed_reading": "",
        "pronunciation_correct": "",
        "error_type": ErrorType.UNCERTAIN.value,
        "error_stage": ErrorStage.UNCERTAIN.value,
        "severity": "",
        "review_status": "PENDING",
        "reviewer_note": "",
        "priority_subset": "",
        "subset_rank": "",
        "subset_score": "",
        "subset_reason": "",
        "review_priority": _priority(categories),
        "candidate_categories": "|".join(sorted(categories)),
        "segment_duration_seconds": segment_audio.get("duration_seconds", ""),
        "chunk_duration_seconds": chunk_audio.get("duration_seconds", ""),
        "rerun_count": "0",
        "reproducible": "",
        "rerun_result": "",
    }


def merge_review_row(
    base: dict[str, Any],
    known: dict[str, str] | None,
    existing: dict[str, str] | None,
) -> dict[str, Any]:
    row = dict(base)
    if known:
        for field in (
            "expected_reading",
            "observed_reading",
            "error_type",
            "error_stage",
            "severity",
            "reproducible",
            "note",
        ):
            if known.get(field, "").strip():
                target = "reviewer_note" if field == "note" else field
                value = known[field].strip()
                row[target] = value.upper() if field == "severity" else value
    if existing:
        for field in HUMAN_EDITABLE_FIELDS:
            if field in existing:
                row[field] = existing[field]
    return row


def _priority_score(row: dict[str, Any], chunk: dict[str, Any]) -> tuple[int, str]:
    weights = {
        "date_range": 120,
        "time_range": 115,
        "currency_per_unit": 110,
        "percentage_range": 105,
        "measurement": 95,
        "date": 90,
        "currency": 85,
        "slash_expression": 80,
        "acronym": 75,
        "hyphen_expression": 70,
        "percentage": 65,
        "decimal": 55,
        "number": 30,
        "english_or_foreign_word": 10,
    }
    categories = set(str(row["candidate_categories"]).split("|")) - {""}
    score = sum(weights.get(category, 0) for category in categories)
    reasons = sorted(categories, key=lambda category: (-weights.get(category, 0), category))
    normalized = str(row["normalized_text"]).lower()
    if "<en>" in normalized:
        score += 50
        reasons.append("embedded_english")
    if not chunk.get("normalizer_idempotent", True):
        score += 60
        reasons.append("non_idempotent_normalization")
    if re.search(r"\b(gpt|cyber|ai|hz|usd|usb|wi[ -]?fi|diasel|model|version)\b", normalized):
        score += 45
        reasons.append("target_technical_token")
    return score, "|".join(dict.fromkeys(reasons))


def _mark_priority_subset(
    rows: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    known_chunk_ids: set[str],
    subset_size: int,
) -> None:
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    ranked = []
    for row in rows:
        if (
            row["chunk_id"] in known_chunk_ids
            or row["review_status"] == "REVIEWED"
        ):
            continue
        score, reason = _priority_score(row, chunks_by_id[row["chunk_id"]])
        ranked.append((score, row["source_file"], row["segment_id"], row["chunk_id"], reason))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    selected = {
        chunk_id: (rank, score, reason)
        for rank, (score, _, _, chunk_id, reason) in enumerate(
            ranked[: max(0, subset_size)], start=1
        )
    }
    for row in rows:
        selection = selected.get(row["chunk_id"])
        if selection:
            rank, score, reason = selection
            row.update(
                {
                    "priority_subset": "YES",
                    "subset_rank": rank,
                    "subset_score": score,
                    "subset_reason": reason,
                }
            )
        else:
            row.update(
                {
                    "priority_subset": "",
                    "subset_rank": "",
                    "subset_score": "",
                    "subset_reason": "",
                }
            )


def validate_review_row(row: dict[str, Any]) -> None:
    status = str(row.get("review_status", "")).strip().upper()
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review_status for {row['chunk_id']}: {status!r}")
    severity = str(row.get("severity", "")).strip().upper()
    if severity not in SEVERITIES:
        raise ValueError(f"Invalid severity for {row['chunk_id']}: {severity!r}")

    error_type = str(row.get("error_type", "")).strip()
    error_stage = str(row.get("error_stage", "")).strip()
    validate_attribution(error_type, error_stage)
    correct = str(row.get("pronunciation_correct", "")).strip().lower()
    if correct not in {"", "true", "false"}:
        raise ValueError(
            f"Invalid pronunciation_correct for {row['chunk_id']}: {correct!r}"
        )
    if error_type == ErrorType.MODEL_ERROR.value and status != "REVIEWED":
        raise ValueError(
            f"MODEL_ERROR cannot be assigned before review is complete: {row['chunk_id']}"
        )
    if status != "REVIEWED":
        return

    if error_type == ErrorType.UNCERTAIN.value:
        raise ValueError(
            f"UNCERTAIN must use NEEDS_RECHECK rather than REVIEWED: {row['chunk_id']}"
        )
    if error_type == ErrorType.NO_ERROR.value and correct != "true":
        raise ValueError(
            f"REVIEWED NO_ERROR requires pronunciation_correct=true: {row['chunk_id']}"
        )
    if error_type in (ErrorType.NORMALIZE_ERROR.value, ErrorType.MODEL_ERROR.value):
        if correct != "false":
            raise ValueError(
                f"REVIEWED {error_type} requires pronunciation_correct=false: "
                f"{row['chunk_id']}"
            )
        for field in ("expected_reading", "observed_reading"):
            if not str(row.get(field, "")).strip():
                raise ValueError(
                    f"REVIEWED {error_type} requires {field}: {row['chunk_id']}"
                )
    if error_type == ErrorType.MODEL_ERROR.value:
        if int(str(row.get("rerun_count", "0") or "0")) < 2:
            raise ValueError(
                f"MODEL_ERROR requires two diagnostic reruns: {row['chunk_id']}"
            )
        if str(row.get("reproducible", "")).strip().lower() not in {
            "true",
            "false",
        }:
            raise ValueError(
                f"MODEL_ERROR requires reproducible=true/false: {row['chunk_id']}"
            )
        if not str(row.get("rerun_result", "")).strip():
            raise ValueError(f"MODEL_ERROR requires rerun_result: {row['chunk_id']}")


def _segment_rows(
    chunks: list[dict[str, Any]],
    segment_audio: dict[str, dict[str, Any]],
    categories: dict[str, set[str]],
    review_rows: list[dict[str, Any]],
    completed_segment_ids: set[str],
    existing_path: Path,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk["segment_id"]].append(chunk)
    existing = {row["segment_id"]: row for row in read_csv(existing_path)}
    selected_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in review_rows:
        if review["priority_subset"] == "YES":
            selected_by_segment[review["segment_id"]].append(review)
    rows = []
    for segment_id, items in grouped.items():
        metadata = segment_audio.get(segment_id, {})
        cats = categories.get(segment_id, set())
        selected = sorted(
            selected_by_segment.get(segment_id, []),
            key=lambda review: int(review["subset_rank"]),
        )
        row: dict[str, Any] = {
            "source_file": items[0]["source_file"],
            "segment_id": segment_id,
            "audio_segment_path": metadata.get("audio_path", ""),
            "raw_text": items[0]["raw_text"],
            "chunk_ids": "|".join(item["chunk_id"] for item in items),
            "chunk_count": len(items),
            "review_priority": _priority(cats),
            "candidate_categories": "|".join(sorted(cats)),
            "duration_seconds": metadata.get("duration_seconds", ""),
            "priority_subset_chunks": "|".join(
                review["chunk_id"] for review in selected
            ),
            "priority_subset_count": len(selected),
            "priority_subset_best_rank": selected[0]["subset_rank"] if selected else "",
            "segment_review_status": "PENDING",
            "suspicious_chunk_ids": "",
            "reviewer_note": "",
        }
        old = existing.get(segment_id)
        if old:
            for field in (
                "segment_review_status",
                "suspicious_chunk_ids",
                "reviewer_note",
            ):
                row[field] = old.get(field, row[field])
        if segment_id in completed_segment_ids:
            row["segment_review_status"] = "REVIEWED"
            confirmation = "Human reviewer confirmed the priority subset pass."
            note = str(row["reviewer_note"]).strip()
            if confirmation not in note:
                row["reviewer_note"] = f"{note} {confirmation}".strip()
        status = str(row["segment_review_status"]).strip().upper()
        if status not in REVIEW_STATUSES:
            raise ValueError(
                f"Invalid segment_review_status for {segment_id}: {status!r}"
            )
        rows.append(row)
    return rows


def build_human_review(
    config_path: Path,
    subset_size: int = 40,
    mark_priority_pass: bool = False,
) -> dict[str, Any]:
    project_dir = config_path.resolve().parent
    config = load_config(config_path)
    output_root = (project_dir / config["outputs"]["root"]).resolve()
    input_paths = [(project_dir / value).resolve() for value in config["inputs"]]
    current_run_id = run_id(config, input_paths)
    chunks = read_jsonl(output_root / "preprocessing" / "preprocessing.jsonl")
    report_root = output_root / "reports"

    review_path = report_root / "human_review.csv"
    segment_path = report_root / "segment_review_index.csv"
    existing = {row["chunk_id"]: row for row in read_csv(review_path)}
    pass_chunk_ids = {
        chunk_id
        for chunk_id, row in existing.items()
        if mark_priority_pass
        and row.get("priority_subset", "").strip().upper() == "YES"
        and row.get("review_status", "").strip().upper() != "REVIEWED"
    }
    if mark_priority_pass and not pass_chunk_ids:
        raise ValueError("No pending priority subset rows are available to mark pass")
    completed_segment_ids = {
        existing[chunk_id]["segment_id"] for chunk_id in pass_chunk_ids
    }
    known = {
        row["chunk_id"]: row
        for row in read_csv(
            (project_dir / config["evaluation"]["preprocessing_findings_file"]).resolve()
        )
    }
    segment_audio = _audio_metadata(output_root, current_run_id, "audio")
    chunk_audio = _audio_metadata(output_root, current_run_id, "chunks")
    segment_categories, chunk_categories = _candidate_categories(report_root, chunks)

    rows = []
    for chunk in chunks:
        row = merge_review_row(
            _base_review_row(
                chunk,
                segment_audio.get(chunk["segment_id"], {}),
                chunk_audio.get(chunk["chunk_id"], {}),
                chunk_categories.get(chunk["chunk_id"], set()),
            ),
            known.get(chunk["chunk_id"]),
            existing.get(chunk["chunk_id"]),
        )
        if chunk["chunk_id"] in pass_chunk_ids:
            confirmation = "Human reviewer confirmed pronunciation pass for priority subset."
            note = str(row["reviewer_note"]).strip()
            row.update(
                {
                    "expected_reading": "",
                    "observed_reading": "",
                    "pronunciation_correct": "true",
                    "error_type": ErrorType.NO_ERROR.value,
                    "error_stage": ErrorStage.NONE.value,
                    "severity": "",
                    "review_status": "REVIEWED",
                    "reviewer_note": (
                        note if confirmation in note else f"{note} {confirmation}".strip()
                    ),
                    "rerun_count": "0",
                    "reproducible": "",
                    "rerun_result": "",
                }
            )
        validate_review_row(row)
        rows.append(row)

    _mark_priority_subset(rows, chunks, set(known), subset_size)
    segment_rows = _segment_rows(
        chunks,
        segment_audio,
        segment_categories,
        rows,
        completed_segment_ids,
        segment_path,
    )
    write_csv(review_path, rows, HUMAN_REVIEW_FIELDS)
    write_csv(segment_path, segment_rows, SEGMENT_REVIEW_FIELDS)
    report_path = report_root / "PHASE2_REPORT.md"
    _write_phase2_report(
        config, output_root, current_run_id, rows, segment_rows, report_path
    )
    return {
        "run_id": current_run_id,
        "total_chunks": len(rows),
        "reviewed_chunks": sum(row["review_status"] == "REVIEWED" for row in rows),
        "priority_subset_chunks": sum(
            row["priority_subset"] == "YES" for row in rows
        ),
        "priority_subset_segments": sum(
            int(row["priority_subset_count"]) > 0 for row in segment_rows
        ),
        "marked_pass": len(pass_chunk_ids),
        "human_review_csv": str(review_path),
        "segment_review_csv": str(segment_path),
        "report": str(report_path),
    }


def _write_phase2_report(
    config: dict[str, Any],
    output_root: Path,
    current_run_id: str,
    rows: list[dict[str, Any]],
    segment_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    manifest = _load_json(output_root / "preprocessing" / "manifest.json")
    reviewed = [row for row in rows if row["review_status"] == "REVIEWED"]
    needs_recheck = [row for row in rows if row["review_status"] == "NEEDS_RECHECK"]
    reviewed_segments = sum(
        row["segment_review_status"] == "REVIEWED" for row in segment_rows
    )
    counts = Counter(row["error_type"] for row in rows)
    g2p_count = sum(
        row["error_type"] == ErrorType.NORMALIZE_ERROR.value
        and row["error_stage"] == ErrorStage.G2P.value
        for row in rows
    )
    reviewed_model_errors = [
        row
        for row in reviewed
        if row["error_type"] == ErrorType.MODEL_ERROR.value
    ]
    reviewed_normalizer_errors = [
        row
        for row in reviewed
        if row["error_type"] == ErrorType.NORMALIZE_ERROR.value
    ]
    carried_normalizer_errors = [
        row
        for row in rows
        if row["error_type"] == ErrorType.NORMALIZE_ERROR.value
        and row["review_status"] != "REVIEWED"
    ]
    coverage = (len(reviewed) / len(rows) * 100) if rows else 0.0
    subset = [row for row in rows if row["priority_subset"] == "YES"]
    subset_reviewed = [row for row in subset if row["review_status"] == "REVIEWED"]
    subset_segments = [
        row for row in segment_rows if int(row["priority_subset_count"]) > 0
    ]
    model_rate = (
        f"{len(reviewed_model_errors) / len(reviewed) * 100:.2f}%"
        if reviewed
        else "N/A (no human-reviewed chunks)"
    )

    lines = [
        "# VieNeu-TTS v3 Turbo - Phase 2 Human Pronunciation Review",
        "",
        "## Baseline",
        "",
        f"- Project status: `{'COMPLETE' if len(reviewed) == len(rows) else 'PHASE2_PARTIAL'}`",
        f"- Run ID: `{current_run_id}`",
        f"- Model: `{config['model']['repo']}`",
        f"- Model revision: `{config['model']['revision']}`",
        f"- VieNeu version: `{manifest['environment']['packages']['vieneu']}`",
        f"- sea-g2p version: `{manifest['environment']['packages']['sea-g2p']}`",
        f"- Backend: `{config['model']['backend']} / {config['model']['precision']} / {config['model']['device']}`",
        f"- Voice: `{config['model']['voice']}`",
        f"- Total segments: {len(segment_rows)}",
        f"- Total chunks: {len(rows)}",
        "",
        "Normalization findings apply specifically to the pinned benchmark environment above.",
        "",
        "## Review Coverage",
        "",
        f"- Segments reviewed: {reviewed_segments}/{len(segment_rows)}",
        f"- Chunks reviewed: {len(reviewed)}/{len(rows)}",
        f"- Pending: {len(rows) - len(reviewed)}",
        f"- Needs recheck: {len(needs_recheck)}",
        f"- Review coverage: {coverage:.2f}%",
        f"- Priority subset: {len(subset)} chunks across {len(subset_segments)} segments",
        f"- Priority subset reviewed: {len(subset_reviewed)}/{len(subset)}",
        "",
        "## Final Error Counts",
        "",
        "These are current classifications. The six Phase 1 normalization findings are carried forward, but remain pending human listening unless listed as reviewed.",
        "",
        f"- NO_ERROR = {counts[ErrorType.NO_ERROR.value]}",
        f"- NORMALIZE_ERROR = {counts[ErrorType.NORMALIZE_ERROR.value]}",
        f"- G2P_ERROR = {g2p_count}",
        f"- MODEL_ERROR = {counts[ErrorType.MODEL_ERROR.value]}",
        f"- UNCERTAIN = {counts[ErrorType.UNCERTAIN.value]}",
        f"- MODEL_ERROR_RATE = {model_rate}",
        "",
        "## Confirmed Model Errors",
        "",
    ]
    if not reviewed_model_errors:
        lines.append("No MODEL_ERROR has been confirmed so far.")
    for index, row in enumerate(reviewed_model_errors, start=1):
        lines.extend(
            [
                f"### Case {index}: {row['chunk_id']}",
                "",
                f"- Raw: {row['raw_text']}",
                f"- Normalized: {row['normalized_text']}",
                f"- Phoneme: {row['phonemes']}",
                f"- Expected: {row['expected_reading']}",
                f"- Observed: {row['observed_reading']}",
                f"- Baseline audio: `{row['audio_chunk_path']}`",
                f"- Rerun result: {row['rerun_result']}",
                f"- Diagnosis: `{row['error_type']} / {row['error_stage']}`",
                f"- Evidence: {row['reviewer_note']}",
                "",
            ]
        )
    lines.extend(["", "## Normalization Errors Confirmed During Listening", ""])
    if not reviewed_normalizer_errors:
        lines.append("None confirmed by human listening yet.")
    for row in reviewed_normalizer_errors:
        lines.append(
            f"- `{row['chunk_id']}`: expected '{row['expected_reading']}', observed "
            f"'{row['observed_reading']}' (`{row['error_stage']}`, {row['severity']})."
        )
    lines.extend(["", "## Phase 1 Findings Awaiting Listening", ""])
    for row in carried_normalizer_errors:
        lines.append(
            f"- `{row['chunk_id']}`: {row['expected_reading']} "
            f"(`{row['error_stage']}`, {row['severity']})."
        )
    lines.extend(
        [
            "",
            "## Uncertain Cases",
            "",
            f"{counts[ErrorType.UNCERTAIN.value]} chunks remain `UNCERTAIN`; "
            "use `segment_review_index.csv` for segment-first listening and "
            "`human_review.csv` for chunk-level attribution.",
            "",
            "ASR triage was not used. Human perceptual judgment remains the ground truth.",
            "",
            "## Conclusion",
            "",
        ]
    )
    if len(reviewed) < len(rows):
        lines.extend(
            [
                "No MODEL_ERROR has been confirmed so far. The remaining unreviewed "
                "samples prevent a complete conclusion.",
                "",
                "PROJECT_STATUS = PHASE2_PARTIAL",
                f"REVIEWED_CHUNKS = {len(reviewed)}",
                f"PENDING_CHUNKS = {len(rows) - len(reviewed)}",
                "BLOCKER = Human perceptual review has not yet been completed.",
                "NEXT_STEP = Listen segment-first, record suspicious chunk IDs, then complete chunk-level attribution.",
            ]
        )
    else:
        lines.append("PROJECT_STATUS = COMPLETE")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable Phase 2 review files")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--subset-size", type=int, default=40)
    parser.add_argument("--mark-priority-pass", action="store_true")
    args = parser.parse_args()
    summary = build_human_review(
        args.config,
        subset_size=args.subset_size,
        mark_priority_pass=args.mark_priority_pass,
    )
    print(
        f"PHASE2_OK run_id={summary['run_id']} chunks={summary['total_chunks']} "
        f"reviewed={summary['reviewed_chunks']} "
        f"priority_subset={summary['priority_subset_chunks']} "
        f"marked_pass={summary['marked_pass']}"
    )


if __name__ == "__main__":
    main()
