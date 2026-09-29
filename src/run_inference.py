from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from vieneu import Vieneu
from vieneu_utils.core_utils import gaps_to_silence, join_audio_chunks

from .utils import (
    load_config,
    read_jsonl,
    run_id,
    sha256_file,
    stable_seed,
    write_json,
)


MODEL_GRAPH_FILES = [
    "vieneu_prefill.onnx",
    "vieneu_decode_step.onnx",
    "vieneu_acoustic_cached.onnx",
    "vieneu_backbone_shared.data",
    "vieneu_v3_heads.npz",
    "config.json",
    "tokenizer.json",
]


def resolve_pinned_model(repo: str, revision: str, subfolder: str) -> tuple[Path, Path]:
    root_config = Path(
        hf_hub_download(repo, "config.json", revision=revision, repo_type="model")
    )
    last = None
    for filename in MODEL_GRAPH_FILES:
        last = hf_hub_download(
            repo,
            filename,
            revision=revision,
            repo_type="model",
            subfolder=subfolder,
        )
    assert last is not None
    return root_config.parent, Path(last).parent


def cached_repo_revision(repo: str) -> str:
    cache_name = "models--" + repo.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
    ref = cache_root / "refs" / "main"
    return ref.read_text(encoding="utf-8").strip() if ref.exists() else "unknown"


def _chunk_paths(output_root: Path, current_run_id: str, row: dict[str, Any]) -> tuple[Path, Path]:
    source_dir = output_root / "chunks" / current_run_id / Path(row["source_file"]).stem
    wav_path = source_dir / f"{row['chunk_id']}.wav"
    return wav_path, wav_path.with_suffix(".json")


def _logical_output_path(
    configured_root: str, output_root: Path, physical_path: Path
) -> str:
    return str(Path(configured_root) / physical_path.relative_to(output_root))


def _load_audio(path: Path) -> np.ndarray:
    wav, _ = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return np.asarray(wav, dtype=np.float32)


def run_inference(
    config_path: Path,
    *,
    resume: bool = True,
    limit_chunks: int | None = None,
    start_chunk: int = 0,
    build_segments: bool = True,
) -> dict[str, Any]:
    project_dir = config_path.resolve().parent
    config = load_config(config_path)
    input_paths = [(project_dir / value).resolve() for value in config["inputs"]]
    output_root = (project_dir / config["outputs"]["root"]).resolve()
    rows = read_jsonl(output_root / "preprocessing" / "preprocessing.jsonl")
    current_run_id = run_id(config, input_paths)
    configured_revision = str(config["model"]["revision"])
    onnx_subfolder = "onnx_int8" if config["model"]["precision"] == "int8" else "onnx_update"
    model_root, onnx_dir = resolve_pinned_model(
        config["model"]["repo"], configured_revision, onnx_subfolder
    )

    tts = Vieneu(
        mode=config["model"]["mode"],
        backbone_repo=str(model_root),
        onnx_dir=str(onnx_dir),
        backend=config["model"]["backend"],
        device=config["model"]["device"],
        precision=config["model"]["precision"],
        threads=int(config["inference"].get("threads", 0)),
    )
    voice_name = config["model"]["voice"]
    speaker_emb, ref_codes = tts._resolve_ref(voice_name, None, True, True)
    sampling = {
        "temperature": float(config["inference"]["temperature"]),
        "top_k": int(config["inference"]["top_k"]),
        "top_p": float(config["inference"]["top_p"]),
        "max_new_frames": int(config["inference"]["max_new_frames"]),
        "repetition_penalty": float(config["inference"]["repetition_penalty"]),
    }
    sample_rate = int(config["inference"]["sample_rate"])
    base_seed = int(config["inference"]["seed"])
    log_path = output_root / "logs" / current_run_id / "inference.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    failed = 0
    stop_chunk = None if limit_chunks is None else start_chunk + limit_chunks
    considered = rows[start_chunk:stop_chunk]
    with log_path.open("a", encoding="utf-8", newline="\n") as log:
        for row in considered:
            wav_path, metadata_path = _chunk_paths(output_root, current_run_id, row)
            if resume and wav_path.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("run_id") == current_run_id and metadata.get("status") == "ok":
                    skipped += 1
                    log.write(json.dumps({"event": "skip", "chunk_id": row["chunk_id"]}) + "\n")
                    continue

            previous_attempts = 0
            if metadata_path.exists():
                try:
                    previous_attempts = int(
                        json.loads(metadata_path.read_text(encoding="utf-8")).get("attempts", 0)
                    )
                except (ValueError, json.JSONDecodeError):
                    previous_attempts = 0
            seed = stable_seed(base_seed, row["chunk_id"])
            started = time.perf_counter()
            try:
                np.random.seed(seed)
                wav = tts._infer_chunks(
                    [row["chunk_normalized_text"]],
                    speaker_emb,
                    ref_codes,
                    config["model"]["style"],
                    True,
                    1,
                    sampling,
                )[0]
                wav = np.asarray(wav, dtype=np.float32)
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav_path, wav, sample_rate)
                elapsed = time.perf_counter() - started
                metadata = {
                    "run_id": current_run_id,
                    "status": "ok",
                    "source_file": row["source_file"],
                    "segment_id": row["segment_id"],
                    "chunk_id": row["chunk_id"],
                    "seed": seed,
                    "attempts": previous_attempts + 1,
                    "elapsed_seconds": round(elapsed, 4),
                    "duration_seconds": round(len(wav) / sample_rate, 4),
                    "sample_rate": sample_rate,
                    "audio_sha256": sha256_file(wav_path),
                    "phonemes": row["phonemes"],
                    "audio_path": _logical_output_path(
                        config["outputs"]["root"], output_root, wav_path
                    ),
                }
                write_json(metadata_path, metadata)
                generated += 1
                log.write(
                    json.dumps(
                        {"event": "generated", "chunk_id": row["chunk_id"], "elapsed": elapsed}
                    )
                    + "\n"
                )
            except Exception as exc:
                failed += 1
                metadata = {
                    "run_id": current_run_id,
                    "status": "failed",
                    "source_file": row["source_file"],
                    "segment_id": row["segment_id"],
                    "chunk_id": row["chunk_id"],
                    "seed": seed,
                    "attempts": previous_attempts + 1,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                }
                write_json(metadata_path, metadata)
                log.write(json.dumps({"event": "failed", **metadata}, ensure_ascii=False) + "\n")

    if not build_segments:
        tts.close()
        worker_manifest = {
            "run_id": current_run_id,
            "start_chunk": start_chunk,
            "chunks_considered": len(considered),
            "chunks_generated": generated,
            "chunks_skipped": skipped,
            "chunks_failed": failed,
            "segment_audio_complete": 0,
            "segment_audio_skipped": 0,
            "segment_audio_incomplete": 0,
        }
        write_json(
            output_root
            / "logs"
            / current_run_id
            / f"inference_worker_{start_chunk:03d}.json",
            worker_manifest,
        )
        return worker_manifest

    # Build segment WAVs only when every constituent chunk exists for this run.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_file"], row["segment_id"])].append(row)

    segment_audio = 0
    segment_audio_skipped = 0
    incomplete_segments = 0
    for (source_file, segment_id), segment_rows in grouped.items():
        segment_rows.sort(key=lambda item: int(item["chunk_index"]))
        paths = [_chunk_paths(output_root, current_run_id, row)[0] for row in segment_rows]
        if not all(path.exists() for path in paths):
            incomplete_segments += 1
            continue
        chunk_hashes = [sha256_file(path) for path in paths]
        segment_path = (
            output_root
            / "audio"
            / current_run_id
            / Path(source_file).stem
            / f"{segment_id}.wav"
        )
        segment_metadata_path = segment_path.with_suffix(".json")
        if resume and segment_path.exists() and segment_metadata_path.exists():
            try:
                existing_segment = json.loads(
                    segment_metadata_path.read_text(encoding="utf-8")
                )
                if (
                    existing_segment.get("run_id") == current_run_id
                    and existing_segment.get("status") == "ok"
                    and existing_segment.get("chunk_audio_sha256") == chunk_hashes
                ):
                    segment_audio_skipped += 1
                    continue
            except (ValueError, json.JSONDecodeError):
                pass
        waveforms = [_load_audio(path) for path in paths]
        gaps = [row["gap_after"] for row in segment_rows[:-1]]
        joined = join_audio_chunks(
            waveforms, sample_rate, silence_ps=gaps_to_silence(gaps)
        )
        if bool(config["inference"].get("apply_watermark", True)):
            joined = tts._apply_watermark(joined)
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(segment_path, joined, sample_rate)
        write_json(
            segment_metadata_path,
            {
                "run_id": current_run_id,
                "status": "ok",
                "source_file": source_file,
                "segment_id": segment_id,
                "chunk_ids": [row["chunk_id"] for row in segment_rows],
                "chunk_audio_sha256": chunk_hashes,
                "sample_rate": sample_rate,
                "duration_seconds": round(len(joined) / sample_rate, 4),
                "audio_sha256": sha256_file(segment_path),
                "audio_path": _logical_output_path(
                    config["outputs"]["root"], output_root, segment_path
                ),
            },
        )
        segment_audio += 1

    tts.close()
    manifest = {
        "run_id": current_run_id,
        "model": config["model"]["repo"],
        "model_revision": configured_revision,
        "codec_repo": config["model"]["codec_repo"],
        "codec_revision": cached_repo_revision(config["model"]["codec_repo"]),
        "backend": config["model"]["backend"],
        "precision": config["model"]["precision"],
        "device": config["model"]["device"],
        "voice": voice_name,
        "sampling": sampling,
        "seed_strategy": f"sha256(chunk_id) + {base_seed}",
        "sample_rate": sample_rate,
        "chunks_total": len(rows),
        "chunks_considered": len(considered),
        "chunks_generated": generated,
        "chunks_skipped": skipped,
        "chunks_failed": failed,
        "segment_audio_complete": segment_audio,
        "segment_audio_skipped": segment_audio_skipped,
        "segment_audio_incomplete": incomplete_segments,
    }
    write_json(output_root / "logs" / current_run_id / "inference_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run resumable VieNeu v3 Turbo inference")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--no-build-segments", action="store_true")
    args = parser.parse_args()
    manifest = run_inference(
        args.config,
        resume=not args.no_resume,
        limit_chunks=args.limit_chunks,
        start_chunk=args.start_chunk,
        build_segments=not args.no_build_segments,
    )
    print(
        f"INFERENCE_DONE run_id={manifest['run_id']} generated={manifest['chunks_generated']} "
        f"skipped={manifest['chunks_skipped']} failed={manifest['chunks_failed']} "
        f"segments={manifest['segment_audio_complete']}"
    )


if __name__ == "__main__":
    main()

