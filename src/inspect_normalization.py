from __future__ import annotations

import argparse
import inspect
import re
from pathlib import Path
from typing import Any

from sea_g2p import G2P
from vieneu.v3turbo import V3TurboVieNeuTTS
from vieneu_utils.phonemize_text import (
    PuncNormalizer,
    RE_NEWLINE_SPLIT,
    normalize_to_chunks_v3_with_gaps,
    phonemize_text_with_emotions,
)

from .schemas import Candidate, ChunkRecord, Segment
from .utils import (
    environment_snapshot,
    load_config,
    package_version,
    read_utf8_preserving_newlines,
    run_id,
    sha256_file,
    write_csv,
    write_json,
    write_jsonl,
)


HEADING_RE = re.compile(r"^Văn bản\s+(\d+)\s*:\s*$", re.MULTILINE)

# Ordered from specific to broad. Overlap is retained because a token may carry
# multiple risks (for example a date range is also a hyphen expression).
CANDIDATE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "date_range",
        re.compile(
            r"(?<!\d)(?:"
            r"\d{1,2}/\d{1,2}(?:/\d{2,4})?\s*[-–—]\s*"
            r"\d{1,2}/\d{1,2}(?:/\d{2,4})?"
            r"|\d{1,2}\s*[-–—]\s*\d{1,2}/\d{1,2}(?:/\d{2,4})?"
            r")(?!\d)"
        ),
    ),
    (
        "date",
        re.compile(r"(?<!\d)\d{1,2}/\d{1,2}(?:/\d{2,4})?(?!\d)"),
    ),
    (
        "time_range",
        re.compile(
            r"(?<!\w)(?:\d{1,2}(?::\d{2}|h\d{0,2})?)\s*[-–—]\s*"
            r"(?:\d{1,2}(?::\d{2}|h\d{0,2})?)(?!\w)",
            re.IGNORECASE,
        ),
    ),
    (
        "time",
        re.compile(r"(?<!\w)\d{1,2}(?::\d{2}|h\d{0,2})(?!\w)", re.IGNORECASE),
    ),
    (
        "percentage_range",
        re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*[-–—]\s*\d+(?:[.,]\d+)?\s*%"),
    ),
    ("percentage", re.compile(r"(?<!\w)\d+(?:[.,]\d+)?\s*%")),
    (
        "currency_per_unit",
        re.compile(
            r"(?<!\w)(?:USD|VND|EUR|GBP|JPY|\$|€|£)\s*/\s*[\wÀ-ỹ]+",
            re.IGNORECASE,
        ),
    ),
    (
        "currency",
        re.compile(
            r"(?:[$€£]\s*\d+(?:[.,]\d+)*|\d+(?:[.,]\d+)*\s*"
            r"(?:USD|VND|EUR|GBP|JPY|euro|riel|đồng))",
            re.IGNORECASE,
        ),
    ),
    (
        "measurement",
        re.compile(
            r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:kg|g|mg|km|m|cm|mm|l|ml|"
            r"Hz|kHz|MHz|GHz|W|kW|MP|mAh|inch|nits)(?!\w)",
            re.IGNORECASE,
        ),
    ),
    (
        "slash_expression",
        re.compile(r"(?<!\w)[A-Za-zÀ-ỹ0-9]+\s*/\s*[A-Za-zÀ-ỹ0-9]+(?!\w)"),
    ),
    (
        "hyphen_expression",
        re.compile(r"(?<!\w)[A-Za-zÀ-ỹ0-9]+(?:-[A-Za-zÀ-ỹ0-9]+)+(?!\w)"),
    ),
    ("decimal", re.compile(r"(?<!\w)\d+[.,]\d+(?!\w)")),
    ("acronym", re.compile(r"(?<!\w)[A-ZĐ]{2,}(?:\d+)?(?!\w)")),
    (
        "english_or_foreign_word",
        re.compile(r"(?<!\w)[A-Za-z][A-Za-z']{2,}(?!\w)"),
    ),
    ("number", re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")),
]

CHUNK_FIELDS = [
    "source_file",
    "segment_id",
    "source_label",
    "chunk_id",
    "chunk_index",
    "raw_text",
    "normalized_text",
    "chunk_raw_text",
    "chunk_normalized_text",
    "renormalized_chunk_text",
    "normalizer_idempotent",
    "phonemes",
    "g2p_only_phonemes",
    "gap_after",
    "source_sha256",
]

CANDIDATE_FIELDS = [
    "source_file",
    "segment_id",
    "category",
    "matched_text",
    "start",
    "end",
    "context",
    "isolated_normalized",
]


def parse_segments(path: Path) -> tuple[list[Segment], dict[str, Any]]:
    text, encoding, has_bom = read_utf8_preserving_newlines(path)
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        raise ValueError(f"No 'Văn bản N:' headings found in {path}")

    source_hash = sha256_file(path)
    segments: list[Segment] = []
    for index, match in enumerate(matches):
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : body_end].strip("\r\n")
        number = int(match.group(1))
        segments.append(
            Segment(
                source_file=path.name,
                segment_id=f"{path.stem}-{number:03d}",
                source_label=f"Văn bản {number}",
                raw_text=body,
                source_sha256=source_hash,
            )
        )

    metadata = {
        "source_file": path.name,
        "absolute_source": str(path.resolve()),
        "encoding": encoding,
        "bom_removed_for_processing": has_bom,
        "bytes": path.stat().st_size,
        "sha256": source_hash,
        "segments": len(segments),
    }
    return segments, metadata


def normalize_exactly_like_v3(segment: Segment, max_chars: int) -> list[ChunkRecord]:
    normalizer = PuncNormalizer()
    g2p = G2P(lang="vi")
    paragraphs = [p for p in RE_NEWLINE_SPLIT.split(segment.raw_text) if p.strip()]
    normalized_text = (
        "\n".join(normalizer.normalize_batch(paragraphs, punc_norm=True))
        if paragraphs
        else ""
    )
    chunks, gaps = normalize_to_chunks_v3_with_gaps(
        segment.raw_text, max_chars=max_chars
    )

    records: list[ChunkRecord] = []
    for index, chunk in enumerate(chunks, start=1):
        # This is the exact function called by V3TurboVieNeuTTS._infer_chunks.
        actual_phonemes = phonemize_text_with_emotions(chunk)
        # Diagnostic control: bypass the second normalizer pass and run G2P only.
        g2p_only = g2p.phonemize_batch([chunk], punc_norm=True)[0]
        renormalized = normalizer.normalize(chunk, punc_norm=True)
        records.append(
            ChunkRecord(
                source_file=segment.source_file,
                segment_id=segment.segment_id,
                source_label=segment.source_label,
                chunk_id=f"{segment.segment_id}-c{index:03d}",
                chunk_index=index,
                raw_text=segment.raw_text,
                normalized_text=normalized_text,
                # VieNeu normalizes before splitting and does not expose a lossless
                # raw-span mapping for each normalized chunk.
                chunk_raw_text="",
                chunk_normalized_text=chunk,
                renormalized_chunk_text=renormalized,
                normalizer_idempotent=renormalized == chunk,
                phonemes=actual_phonemes,
                g2p_only_phonemes=g2p_only,
                gap_after=gaps[index - 1] if index <= len(gaps) else "",
                source_sha256=segment.source_sha256,
            )
        )
    return records


def scan_candidates(segment: Segment) -> list[Candidate]:
    normalizer = PuncNormalizer()
    rows: list[Candidate] = []
    seen: set[tuple[str, int, int]] = set()
    for category, pattern in CANDIDATE_PATTERNS:
        for match in pattern.finditer(segment.raw_text):
            key = (category, match.start(), match.end())
            if key in seen:
                continue
            seen.add(key)
            left = max(0, match.start() - 45)
            right = min(len(segment.raw_text), match.end() + 45)
            context = segment.raw_text[left:right].replace("\r", " ").replace("\n", " ")
            rows.append(
                Candidate(
                    source_file=segment.source_file,
                    segment_id=segment.segment_id,
                    category=category,
                    matched_text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    context=context,
                    isolated_normalized=normalizer.normalize(
                        match.group(0), punc_norm=True
                    ),
                )
            )
    return rows


def _source_location(obj: Any) -> dict[str, Any]:
    source_file = inspect.getsourcefile(obj)
    _, line = inspect.getsourcelines(obj)
    return {"file": source_file, "line": line, "symbol": obj.__qualname__}


def pipeline_audit() -> dict[str, Any]:
    from sea_g2p.pipeline import SEAPipeline
    from vieneu_utils.phonemize_text import normalize_to_chunks_v3_with_gaps

    return {
        "verified_versions": {
            "vieneu": package_version("vieneu"),
            "sea-g2p": package_version("sea-g2p"),
        },
        "order": [
            "raw paragraphs",
            "sea_g2p.Normalizer(punc_norm=True)",
            "split normalized text into chunks",
            "phonemize_text_with_emotions (SEAPipeline: normalize again, then G2P)",
            "ONNX generation",
            "join chunk waveforms with boundary-specific silence",
            "watermark segment waveform",
        ],
        "source_locations": {
            "v3_infer": _source_location(V3TurboVieNeuTTS.infer),
            "v3_chunk_generation": _source_location(V3TurboVieNeuTTS._infer_chunks),
            "normalize_and_chunk": _source_location(normalize_to_chunks_v3_with_gaps),
            "sea_pipeline": _source_location(SEAPipeline.run),
        },
        "findings": {
            "chunking": "after paragraph-level normalization",
            "punctuation": "punc_norm=True in normalizer and actual phonemizer",
            "skip_normalize": (
                "available in vieneu_utils.phonemize_batch/phonemize_with_dict, "
                "but not exposed by the v3turbo infer path"
            ),
            "double_normalization": (
                "v3 chunks are already normalized, then the actual phonemizer "
                "runs SEAPipeline and normalizes each chunk again"
            ),
            "mixed_language": "delegated to sea-g2p Vietnamese/English pipeline",
        },
    }


def run_preprocessing(config_path: Path) -> dict[str, Any]:
    project_dir = config_path.resolve().parent
    config = load_config(config_path)
    input_paths = [(project_dir / value).resolve() for value in config["inputs"]]
    output_root = (project_dir / config["outputs"]["root"]).resolve()
    preprocessing_dir = output_root / "preprocessing"
    report_dir = output_root / "reports"
    current_run_id = run_id(config, input_paths)

    segments: list[Segment] = []
    input_metadata: list[dict[str, Any]] = []
    for path in input_paths:
        parsed, metadata = parse_segments(path)
        segments.extend(parsed)
        input_metadata.append(metadata)

    chunks: list[ChunkRecord] = []
    candidates: list[Candidate] = []
    max_chars = int(config["inference"]["max_chars"])
    for segment in segments:
        chunks.extend(normalize_exactly_like_v3(segment, max_chars=max_chars))
        candidates.extend(scan_candidates(segment))

    chunk_rows = [row.to_dict() for row in chunks]
    candidate_rows = [row.to_dict() for row in candidates]
    write_jsonl(preprocessing_dir / "preprocessing.jsonl", chunk_rows)
    write_csv(preprocessing_dir / "preprocessing.csv", chunk_rows, CHUNK_FIELDS)
    write_csv(report_dir / "special_cases.csv", candidate_rows, CANDIDATE_FIELDS)

    manifest = {
        "run_id": current_run_id,
        "inputs": input_metadata,
        "total_segments": len(segments),
        "total_chunks": len(chunks),
        "total_candidates": len(candidates),
        "non_idempotent_chunks": sum(not row.normalizer_idempotent for row in chunks),
        "environment": environment_snapshot(),
        "pipeline": pipeline_audit(),
    }
    write_json(preprocessing_dir / "manifest.json", manifest)
    write_json(report_dir / "pipeline_audit.json", manifest["pipeline"])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump VieNeu normalization and G2P")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    args = parser.parse_args()
    manifest = run_preprocessing(args.config)
    print(
        f"PREPROCESSING_OK run_id={manifest['run_id']} "
        f"segments={manifest['total_segments']} chunks={manifest['total_chunks']} "
        f"candidates={manifest['total_candidates']}"
    )


if __name__ == "__main__":
    main()

