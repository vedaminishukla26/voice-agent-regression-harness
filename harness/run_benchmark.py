"""Harness entry point.

Runs one phase at a time. Phases beyond ``tts`` are not implemented yet and say
so plainly rather than failing in some more creative way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from persona_spec import PERSONAS
from persona_tts import (
    DEFAULT_OUTPUT_DIR,
    CartesiaTTSBackend,
    DryRunBackend,
    TTSBackend,
    TTSError,
    render_persona_batch,
    write_results,
)

import os

PHASES = ("tts", "audio", "metrics", "gates", "full")


def run_tts_phase(
    backend: TTSBackend, out_dir: Path, personas: Optional[List[str]] = None
) -> int:
    """Render every persona's script. Returns a process exit code."""
    targets = personas or sorted(PERSONAS)
    results = []
    for name in targets:
        print(f"\n=== {name} ===")
        results.extend(render_persona_batch(name, backend, out_dir))

    summary = write_results(results, out_dir, "benchmark_tts")
    ok = sum(1 for r in results if r.ok)
    failed = len(results) - ok
    print(f"\nTTS phase: {ok} rendered, {failed} failed. Summary: {summary}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark", description="Run a harness phase."
    )
    parser.add_argument("--mode", choices=PHASES, default="tts")
    parser.add_argument("--dry-run", action="store_true", help="no network, no audio")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--persona", action="append", choices=sorted(PERSONAS))
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode in ("audio", "metrics", "gates", "full"):
        print(
            f"Phase {args.mode!r} is not implemented yet.\n"
            "  audio   -- week 2, needs a running media server and an agent\n"
            "  metrics -- week 3, needs captured sessions\n"
            "  gates   -- week 4, needs captured transcripts",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        backend: TTSBackend = DryRunBackend()
    else:
        if not os.environ.get("CARTESIA_API_KEY"):
            print(
                "No TTS API key found. Re-run with --dry-run to exercise the "
                "pipeline offline, or set CARTESIA_API_KEY.",
                file=sys.stderr,
            )
            return 2
        try:
            backend = CartesiaTTSBackend(
                api_key=os.environ["CARTESIA_API_KEY"],
                voice_id=os.environ.get("CARTESIA_VOICE_ID", ""),
            )
        except TTSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    return run_tts_phase(backend, args.output_dir, args.persona)


if __name__ == "__main__":
    raise SystemExit(main())
