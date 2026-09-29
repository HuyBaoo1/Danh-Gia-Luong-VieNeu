from __future__ import annotations

import argparse
from pathlib import Path

from src.diagnose import build_error_analysis
from src.human_review import build_human_review
from src.inspect_normalization import run_preprocessing
from src.run_inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="VieNeu v3 Turbo diagnostic evaluator")
    parser.add_argument(
        "--phase",
        choices=("preprocess", "infer", "report", "phase2", "all"),
        default="all",
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit-chunks", type=int)
    parser.add_argument("--start-chunk", type=int, default=0)
    parser.add_argument("--no-build-segments", action="store_true")
    parser.add_argument("--subset-size", type=int, default=40)
    parser.add_argument("--mark-priority-pass", action="store_true")
    args = parser.parse_args()

    if args.phase in ("preprocess", "all"):
        manifest = run_preprocessing(args.config)
        print(
            f"STAGE=preprocess STATUS=PASS segments={manifest['total_segments']} "
            f"chunks={manifest['total_chunks']} candidates={manifest['total_candidates']}"
        )
    if args.phase in ("infer", "all"):
        manifest = run_inference(
            args.config,
            resume=not args.no_resume,
            limit_chunks=args.limit_chunks,
            start_chunk=args.start_chunk,
            build_segments=not args.no_build_segments,
        )
        status = "PASS" if manifest["chunks_failed"] == 0 else "FAIL"
        print(
            f"STAGE=infer STATUS={status} generated={manifest['chunks_generated']} "
            f"skipped={manifest['chunks_skipped']} failed={manifest['chunks_failed']}"
        )
    if args.phase in ("report", "all"):
        summary = build_error_analysis(args.config)
        print(
            f"STAGE=report STATUS=PASS chunks={summary['total_chunks']} "
            f"counts={summary['counts']}"
        )
    if args.phase == "phase2":
        summary = build_human_review(
            args.config,
            subset_size=args.subset_size,
            mark_priority_pass=args.mark_priority_pass,
        )
        print(
            f"STAGE=phase2 STATUS=PASS chunks={summary['total_chunks']} "
            f"reviewed={summary['reviewed_chunks']} "
            f"priority_subset={summary['priority_subset_chunks']} "
            f"marked_pass={summary['marked_pass']}"
        )


if __name__ == "__main__":
    main()

