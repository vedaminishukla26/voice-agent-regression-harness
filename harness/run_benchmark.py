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
from transport import TransportError

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


def run_audio_phase(
    personas: Optional[List[str]],
    room: str,
    transport: str,
    clock: str,
    max_turns: int,
    max_seconds: int,
    seed: int,
    use_tts: bool,
) -> int:
    """Run one audio session per persona and report where each one stopped.

    Sessions run in sequence rather than concurrently. Against a real room they
    would otherwise contend for the same agent, and the latency numbers this
    exists to produce would measure the contention instead of the agent.
    """
    import asyncio

    from audio_loop import (
        DEFAULT_AUDIO_DIR,
        DEFAULT_TRANSCRIPT_DIR,
        SessionConfig,
        run_session,
    )
    from session_log import STOP_SCRIPT_EXHAUSTED, STOP_TURN_CEILING

    targets = personas or sorted(PERSONAS)
    healthy = {STOP_SCRIPT_EXHAUSTED, STOP_TURN_CEILING}
    summaries = []

    for name in targets:
        print(f"\n=== {name} ===")
        config = SessionConfig(
            persona=name,
            room=room,
            max_turns=max_turns,
            max_wall_clock_ms=max_seconds * 1000,
            seed=seed,
        )
        try:
            log = asyncio.run(
                run_session(
                    config,
                    transport_kind=transport,
                    clock=clock,
                    use_tts=use_tts,
                    transcript_dir=DEFAULT_TRANSCRIPT_DIR,
                    audio_dir=DEFAULT_AUDIO_DIR,
                )
            )
        except TransportError as exc:
            print(f"  transport error: {exc}", file=sys.stderr)
            return 2
        summaries.append(log)

    print("\nAudio phase")
    for log in summaries:
        print("  " + log.summary_line())
    failed = [s for s in summaries if s.stop_reason not in healthy]
    print(f"\n{len(summaries) - len(failed)} healthy, {len(failed)} ended early.")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_benchmark", description="Run a harness phase."
    )
    parser.add_argument("--mode", choices=PHASES, default="tts")
    parser.add_argument("--dry-run", action="store_true", help="no network, no audio")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--persona", action="append", choices=sorted(PERSONAS))

    audio = parser.add_argument_group("audio phase")
    audio.add_argument("--room", default="harness")
    audio.add_argument(
        "--transport",
        choices=("loopback", "livekit"),
        default="loopback",
        help="loopback simulates an agent in-process and needs no server",
    )
    audio.add_argument(
        "--clock",
        choices=("real", "virtual"),
        default="real",
        help="virtual simulates time instead of spending it (loopback only)",
    )
    audio.add_argument("--max-turns", type=int, default=12)
    audio.add_argument("--max-seconds", type=int, default=180)
    audio.add_argument("--seed", type=int, default=0)
    audio.add_argument(
        "--tts", action="store_true", help="render real speech rather than shaped tones"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "audio":
        try:
            return run_audio_phase(
                personas=args.persona,
                room=args.room,
                transport=args.transport,
                clock=args.clock,
                max_turns=args.max_turns,
                max_seconds=args.max_seconds,
                seed=args.seed,
                use_tts=args.tts,
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    if args.mode in ("metrics", "gates", "full"):
        print(
            f"Phase {args.mode!r} is not implemented yet.\n"
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
