from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .schemas import ErrorStage, ErrorType, validate_attribution
from .utils import load_config, read_csv, read_jsonl, run_id, write_csv


REVIEW_FIELDS = [
    "source_file",
    "segment_id",
    "chunk_id",
    "expected_reading",
    "observed_reading",
    "error_type",
    "error_stage",
    "severity",
    "reproducible",
    "note",
    "proof",
]

ERROR_FIELDS = [
    "source_file",
    "segment_id",
    "chunk_id",
    "raw_text",
    "normalized_text",
    "phonemes",
    "expected_reading",
    "observed_reading",
    "error_type",
    "error_stage",
    "severity",
    "audio_path",
    "rerun_count",
    "reproducible",
    "note",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_review_template(
    path: Path,
    chunks: list[dict[str, Any]],
    known_findings: list[dict[str, str]],
) -> None:
    existing = {row.get("chunk_id", ""): row for row in read_csv(path)}
    known = {row.get("chunk_id", ""): row for row in known_findings}
    rows = []
    for chunk in chunks:
        row = {field: "" for field in REVIEW_FIELDS}
        row.update(
            {
                "source_file": chunk["source_file"],
                "segment_id": chunk["segment_id"],
                "chunk_id": chunk["chunk_id"],
            }
        )
        if chunk["chunk_id"] in known:
            row.update(known[chunk["chunk_id"]])
        if chunk["chunk_id"] in existing:
            row.update(
                {
                    key: value
                    for key, value in existing[chunk["chunk_id"]].items()
                    if value.strip()
                }
            )
        rows.append(row)
    write_csv(path, rows, REVIEW_FIELDS)


def _audio_metadata(
    output_root: Path, current_run_id: str, chunk: dict[str, Any]
) -> tuple[str, int, bool]:
    base = (
        output_root
        / "chunks"
        / current_run_id
        / Path(chunk["source_file"]).stem
        / f"{chunk['chunk_id']}"
    )
    wav_path = base.with_suffix(".wav")
    metadata_path = base.with_suffix(".json")
    if not wav_path.exists() or not metadata_path.exists():
        return "", 0, False
    metadata = _load_json(metadata_path)
    if metadata.get("status") != "ok" or metadata.get("run_id") != current_run_id:
        return "", max(0, int(metadata.get("attempts", 0)) - 1), False
    return (
        metadata.get("audio_path", str(wav_path)),
        max(0, int(metadata.get("attempts", 1)) - 1),
        True,
    )


def _reviewed_attribution(
    review: dict[str, str], rerun_count: int
) -> tuple[str, str]:
    error_type = review.get("error_type", "").strip()
    error_stage = review.get("error_stage", "").strip()
    if not error_type and not error_stage:
        return ErrorType.UNCERTAIN.value, ErrorStage.UNCERTAIN.value
    validate_attribution(error_type, error_stage)
    if error_type == ErrorType.MODEL_ERROR.value:
        if not review.get("expected_reading", "").strip():
            raise ValueError(f"MODEL_ERROR requires expected_reading: {review['chunk_id']}")
        if not review.get("observed_reading", "").strip():
            raise ValueError(f"MODEL_ERROR requires observed_reading: {review['chunk_id']}")
        if rerun_count < 1 or review.get("reproducible", "").strip().lower() != "true":
            raise ValueError(
                f"MODEL_ERROR requires at least one rerun and reproducible=true: "
                f"{review['chunk_id']}"
            )
    if error_type == ErrorType.NORMALIZE_ERROR.value and not review.get(
        "expected_reading", ""
    ).strip():
        raise ValueError(
            f"NORMALIZE_ERROR requires expected_reading: {review['chunk_id']}"
        )
    return error_type, error_stage


def build_error_analysis(config_path: Path) -> dict[str, Any]:
    project_dir = config_path.resolve().parent
    config = load_config(config_path)
    input_paths = [(project_dir / value).resolve() for value in config["inputs"]]
    output_root = (project_dir / config["outputs"]["root"]).resolve()
    current_run_id = run_id(config, input_paths)
    chunks = read_jsonl(output_root / "preprocessing" / "preprocessing.jsonl")
    review_path = (project_dir / config["evaluation"]["review_file"]).resolve()
    findings_path = (
        project_dir / config["evaluation"]["preprocessing_findings_file"]
    ).resolve()
    ensure_review_template(review_path, chunks, read_csv(findings_path))
    reviews = {row["chunk_id"]: row for row in read_csv(review_path)}

    analysis_rows: list[dict[str, Any]] = []
    for chunk in chunks:
        review = reviews[chunk["chunk_id"]]
        audio_path, rerun_count, audio_ok = _audio_metadata(
            output_root, current_run_id, chunk
        )
        error_type, error_stage = _reviewed_attribution(review, rerun_count)
        note = review.get("note", "").strip()
        if error_type == ErrorType.UNCERTAIN.value and not note:
            note = (
                "Pending human audio review; preprocessing evidence is retained."
                if audio_ok
                else "Audio is not available for this chunk."
            )
        proof = review.get("proof", "").strip()
        if proof:
            note = f"{note} Proof: {proof}".strip()
        analysis_rows.append(
            {
                "source_file": chunk["source_file"],
                "segment_id": chunk["segment_id"],
                "chunk_id": chunk["chunk_id"],
                "raw_text": chunk["raw_text"],
                "normalized_text": chunk["chunk_normalized_text"],
                "phonemes": chunk["phonemes"],
                "expected_reading": review.get("expected_reading", ""),
                "observed_reading": review.get("observed_reading", ""),
                "error_type": error_type,
                "error_stage": error_stage,
                "severity": review.get("severity", "") or "review_required",
                "audio_path": audio_path,
                "rerun_count": rerun_count,
                "reproducible": review.get("reproducible", ""),
                "note": note,
            }
        )

    versioned_dir = output_root / "reports" / current_run_id
    versioned_csv = versioned_dir / "error_analysis.csv"
    stable_csv = output_root / "reports" / "error_analysis.csv"
    write_csv(versioned_csv, analysis_rows, ERROR_FIELDS)
    write_csv(stable_csv, analysis_rows, ERROR_FIELDS)

    summary = _write_report(
        config, output_root, current_run_id, chunks, analysis_rows, versioned_dir
    )
    shutil.copyfile(versioned_dir / "REPORT.md", output_root / "reports" / "REPORT.md")
    return summary


def _write_report(
    config: dict[str, Any],
    output_root: Path,
    current_run_id: str,
    chunks: list[dict[str, Any]],
    analysis_rows: list[dict[str, Any]],
    versioned_dir: Path,
) -> dict[str, Any]:
    preprocess_manifest = _load_json(output_root / "preprocessing" / "manifest.json")
    inference_path = output_root / "logs" / current_run_id / "inference_manifest.json"
    inference = _load_json(inference_path) if inference_path.exists() else {}
    counts = Counter(row["error_type"] for row in analysis_rows)
    stages = Counter(
        row["error_stage"]
        for row in analysis_rows
        if row["error_type"] == ErrorType.NORMALIZE_ERROR.value
    )
    per_source = defaultdict(int)
    for chunk in chunks:
        per_source[chunk["source_file"]] += 1

    important = [
        row
        for row in analysis_rows
        if row["error_type"]
        in (ErrorType.NORMALIZE_ERROR.value, ErrorType.MODEL_ERROR.value)
    ]
    generated_audio_count = sum(bool(row["audio_path"]) for row in analysis_rows)
    lines = [
        "# VieNeu-TTS v3 Turbo Normalization Evaluation",
        "",
        "## Environment",
        "",
        f"- Run ID: `{current_run_id}`",
        f"- Python: `{preprocess_manifest['environment']['python']}`",
        f"- OS: `{preprocess_manifest['environment']['os']}`",
        f"- CPU: `{preprocess_manifest['environment']['cpu']}`",
        f"- RAM: `{preprocess_manifest['environment'].get('ram_gb', 'unknown')} GB`",
        f"- GPU: `{preprocess_manifest['environment']['gpu']}`",
        f"- Backend: `{config['model']['backend']}` / `{config['model']['precision']}` / `{config['model']['device']}`",
        f"- `vieneu`: `{preprocess_manifest['environment']['packages']['vieneu']}`",
        f"- `sea-g2p`: `{preprocess_manifest['environment']['packages']['sea-g2p']}`",
        f"- `onnxruntime`: `{preprocess_manifest['environment']['packages']['onnxruntime']}`",
        "",
        "## Pipeline verified",
        "",
        "Installed source confirms: paragraph normalization with `punc_norm=True`, "
        "then normalized-text chunking, then `phonemize_text_with_emotions`, ONNX "
        "generation, boundary-aware joining, and segment watermarking.",
        "",
        "The v3 path normalizes each chunk again inside the SEA pipeline. The dump "
        "therefore includes the first normalized chunk, second-pass text, actual "
        "phonemes, and a G2P-only control.",
        "",
        "## Dataset",
        "",
    ]
    for item in preprocess_manifest["inputs"]:
        lines.append(
            f"- `{item['source_file']}`: {item['segments']} segments, "
            f"{per_source[item['source_file']]} chunks, SHA-256 `{item['sha256']}`"
        )
    lines.extend(
        [
            f"- Total segments: {preprocess_manifest['total_segments']}",
            f"- Total chunks: {preprocess_manifest['total_chunks']}",
            f"- Special-case candidates: {preprocess_manifest['total_candidates']}",
            "",
            "## Results",
            "",
            f"- Generated chunks: {generated_audio_count}",
            f"- Failed chunks: {inference.get('chunks_failed', 'not run')}",
            f"- NORMALIZE_ERROR: {counts[ErrorType.NORMALIZE_ERROR.value]}",
            f"- normalizer: {stages[ErrorStage.NORMALIZER.value]}",
            f"- g2p: {stages[ErrorStage.G2P.value]}",
            f"- MODEL_ERROR: {counts[ErrorType.MODEL_ERROR.value]}",
            f"- UNCERTAIN: {counts[ErrorType.UNCERTAIN.value]}",
            f"- NO_ERROR: {counts[ErrorType.NO_ERROR.value]}",
            "",
            "Generation-only attribution remains `UNCERTAIN` until a human records "
            "the observed pronunciation. ASR is not used as ground truth.",
            "",
            "## Important cases",
            "",
        ]
    )
    if not important:
        lines.append("No case has sufficient evidence for a confirmed error yet.")
    for row in important[:20]:
        review = row["raw_text"].replace("\r", " ").replace("\n", " ")
        if len(review) > 260:
            review = review[:257].rstrip() + "..."
        lines.extend(
            [
                f"### {row['chunk_id']}",
                "",
                f"- Raw: {review}",
                f"- Normalized: {row['normalized_text']}",
                f"- Phoneme: {row['phonemes']}",
                f"- Expected: {row['expected_reading']}",
                f"- Observed: {row['observed_reading'] or 'not audio-reviewed'}",
                f"- Diagnosis: `{row['error_type']} / {row['error_stage']}`",
                f"- Proof: {row['note']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            "",
            "Confirmed preprocessing errors are reported separately from generation "
            "errors. Any generated audio that has not been listened to remains "
            "`UNCERTAIN`; this is deliberate evidence discipline, not a missing sample.",
            "",
        ]
    )
    versioned_dir.mkdir(parents=True, exist_ok=True)
    (versioned_dir / "REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    return {
        "run_id": current_run_id,
        "total_chunks": len(chunks),
        "counts": dict(counts),
        "report": str(versioned_dir / "REPORT.md"),
        "error_csv": str(versioned_dir / "error_analysis.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-based error report")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    summary = build_error_analysis(args.config)
    print(
        f"REPORT_OK run_id={summary['run_id']} chunks={summary['total_chunks']} "
        f"counts={summary['counts']}"
    )


if __name__ == "__main__":
    main()

