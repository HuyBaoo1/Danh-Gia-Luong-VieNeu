from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class ErrorType(str, Enum):
    NORMALIZE_ERROR = "NORMALIZE_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    UNCERTAIN = "UNCERTAIN"
    NO_ERROR = "NO_ERROR"


class ErrorStage(str, Enum):
    NORMALIZER = "normalizer"
    G2P = "g2p"
    GENERATION = "generation"
    NONE = "none"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class Segment:
    source_file: str
    segment_id: str
    source_label: str
    raw_text: str
    source_sha256: str


@dataclass(frozen=True)
class ChunkRecord:
    source_file: str
    segment_id: str
    source_label: str
    chunk_id: str
    chunk_index: int
    raw_text: str
    normalized_text: str
    chunk_raw_text: str
    chunk_normalized_text: str
    renormalized_chunk_text: str
    normalizer_idempotent: bool
    phonemes: str
    g2p_only_phonemes: str
    gap_after: str
    source_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Candidate:
    source_file: str
    segment_id: str
    category: str
    matched_text: str
    start: int
    end: int
    context: str
    isolated_normalized: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VALID_ATTRIBUTIONS = {
    (ErrorType.NORMALIZE_ERROR.value, ErrorStage.NORMALIZER.value),
    (ErrorType.NORMALIZE_ERROR.value, ErrorStage.G2P.value),
    (ErrorType.MODEL_ERROR.value, ErrorStage.GENERATION.value),
    (ErrorType.UNCERTAIN.value, ErrorStage.UNCERTAIN.value),
    (ErrorType.NO_ERROR.value, ErrorStage.NONE.value),
}


def validate_attribution(error_type: str, error_stage: str) -> None:
    if (error_type, error_stage) not in VALID_ATTRIBUTIONS:
        raise ValueError(
            f"Invalid attribution: error_type={error_type!r}, "
            f"error_stage={error_stage!r}"
        )

