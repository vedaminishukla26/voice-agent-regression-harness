"""Render candidate script turns to audio.

Week 1 of the harness. The only goal here is to turn a persona's text into a
real audio file, correctly reflecting that persona's delivery settings. No
LiveKit, no agent, no measurement yet.

Two deliberate choices:

*Backends are pluggable.* ``TTSBackend`` is a tiny protocol, so a hosted
provider and a fully local engine are interchangeable. That matters because a
hosted provider means an account and billing, and the harness should be able to
run without one.

*No vendor SDK.* The hosted backend speaks HTTP over the standard library
instead of importing a provider SDK. It keeps the dependency surface at zero for
this layer, and it means the module imports on any Python without waiting for a
vendor to publish wheels for a new interpreter version.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from audio_io import estimate_duration_ms, synthesize_speech_like
from candidate_scripts import CandidateTurn, get_script
from persona_spec import PERSONAS, PersonaSpec, get_persona

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "audio"

CARTESIA_ENDPOINT = "https://api.cartesia.ai/tts/bytes"
# Cartesia pins behaviour to a dated API version header rather than a URL path.
CARTESIA_API_VERSION = "2024-11-13"
DEFAULT_MODEL_ID = "sonic-3"
REQUEST_TIMEOUT_SECONDS = 60

# Compressed output, for listening to by ear.
MP3_OUTPUT_FORMAT = {"container": "mp3", "bit_rate": 128_000, "sample_rate": 44_100}


def pcm_output_format(sample_rate: int) -> Dict:
    """Raw PCM at the audio loop's native rate -- nothing to decode or resample."""
    return {"container": "raw", "encoding": "pcm_s16le", "sample_rate": sample_rate}


class TTSError(RuntimeError):
    """Raised when a backend cannot produce audio for a turn."""


@dataclass
class RenderResult:
    """Outcome of rendering one turn. Failures are recorded, not raised."""

    persona: str
    index: int
    intent: str
    text: str
    ok: bool
    path: Optional[str] = None
    bytes_written: int = 0
    duration_ms: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class TTSBackend(Protocol):
    """Anything that can turn text plus a persona into audio bytes."""

    #: File extension the backend produces, without the leading dot.
    extension: str

    def synthesize(self, text: str, persona: PersonaSpec) -> bytes: ...

    def synthesize_pcm(
        self, text: str, persona: PersonaSpec, sample_rate: int
    ) -> bytes: ...


class DryRunBackend:
    """Produces no audio and makes no network calls.

    Exists so the whole pipeline -- argument parsing, persona resolution, output
    paths, result accounting -- can be exercised with no account, no key and no
    billing. This is the default, so running the module by accident costs
    nothing.
    """

    extension = "txt"

    def __init__(self) -> None:
        self.calls: List[str] = []

    def synthesize(self, text: str, persona: PersonaSpec) -> bytes:
        self.calls.append(text)
        preview = (
            f"[dry-run] persona={persona.name} lang={persona.language} "
            f"rate={persona.speech_rate} text={text!r}\n"
        )
        return preview.encode("utf-8")

    def synthesize_pcm(
        self, text: str, persona: PersonaSpec, sample_rate: int
    ) -> bytes:
        """Shaped tones of the right duration, for exercising the audio loop.

        Not speech and not pretending to be. The loop measures *when* audio
        starts and stops, so an utterance of the right length with a realistic
        energy envelope puts every timing path under the same pressure real
        speech would. Words would only matter to a layer that transcribes the
        candidate, and nothing here does.
        """
        self.calls.append(text)
        # crc32 rather than hash(): string hashing is salted per process, so
        # hash() would give this "deterministic" backend different audio on
        # every run, which is precisely the property a regression harness
        # cannot afford.
        seed = zlib.crc32(f"{persona.name}:{text}".encode("utf-8")) % 10_000
        return synthesize_speech_like(
            estimate_duration_ms(text, persona.speech_rate),
            seed=seed,
            sample_rate=sample_rate,
        ).data


class CartesiaTTSBackend:
    """Hosted TTS over Cartesia's HTTP API.

    Requires a personal API key. Nothing about this backend is load-bearing for
    the harness design -- it is one implementation of :class:`TTSBackend`, and a
    local engine can replace it without touching anything above.
    """

    extension = "mp3"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = DEFAULT_MODEL_ID,
        endpoint: str = CARTESIA_ENDPOINT,
    ) -> None:
        if not api_key:
            raise TTSError(
                "no API key. Set CARTESIA_API_KEY or pass --api-key. "
                "Use --dry-run to exercise the pipeline without one."
            )
        if not voice_id:
            raise TTSError(
                "no voice id. Set CARTESIA_VOICE_ID or pass --voice-id. "
                "Voice ids come from your provider dashboard; this project "
                "ships none so that it carries no account-specific values."
            )
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.endpoint = endpoint

    def build_payload(
        self, text: str, persona: PersonaSpec, output_format: Optional[Dict] = None
    ) -> Dict:
        """Construct the request body.

        Split out from :meth:`synthesize` so payload shaping is testable without
        a key or a network, and so the one place that encodes provider-specific
        field names is easy to find when the API moves.
        """
        return {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {"mode": "id", "id": persona.voice_id or self.voice_id},
            "language": persona.language,
            "speed": persona.speech_rate,
            "output_format": output_format or MP3_OUTPUT_FORMAT,
        }

    def synthesize_pcm(
        self, text: str, persona: PersonaSpec, sample_rate: int
    ) -> bytes:
        """Synthesise straight to raw PCM at the transport's sample rate.

        Asking the provider for raw PCM is what keeps an MP3 decoder and a
        resampler out of this project entirely. Both are avoidable simply by
        requesting the format the audio loop already wants, and neither is a
        dependency worth carrying to undo a conversion nobody asked for.
        """
        return self._post(self.build_payload(text, persona, pcm_output_format(sample_rate)))

    def synthesize(self, text: str, persona: PersonaSpec) -> bytes:
        return self._post(self.build_payload(text, persona))

    def _post(self, payload: Dict) -> bytes:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Cartesia-Version": CARTESIA_API_VERSION,
                "X-API-Key": self.api_key,
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=REQUEST_TIMEOUT_SECONDS
            ) as response:
                audio = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise TTSError(f"HTTP {exc.code} from TTS provider: {detail}") from exc
        except urllib.error.URLError as exc:
            raise TTSError(f"could not reach TTS provider: {exc.reason}") from exc

        if not audio:
            raise TTSError("provider returned an empty audio body")
        return audio


def slugify(text: str, limit: int = 32) -> str:
    """Filename-safe fragment of a turn, so output files are self-describing."""
    keep = [c.lower() if c.isalnum() else "_" for c in text.strip()]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:limit] or "turn"


def output_path_for(
    persona: PersonaSpec, index: int, turn: CandidateTurn, out_dir: Path, ext: str
) -> Path:
    """Stable, sortable, human-readable path for one rendered turn."""
    return out_dir / f"{persona.name}_{index:02d}_{turn.intent}_{slugify(turn.text)}.{ext}"


def render_turn(
    backend: TTSBackend,
    persona: PersonaSpec,
    turn: CandidateTurn,
    index: int,
    out_dir: Path,
) -> RenderResult:
    """Render one turn, recording failure rather than raising.

    A single bad turn should never abandon a batch: the failure is more useful
    recorded next to the successes than as a traceback that loses them.
    """
    path = output_path_for(persona, index, turn, out_dir, backend.extension)
    started = time.monotonic()
    try:
        audio = backend.synthesize(turn.text, persona)
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
    except (TTSError, OSError) as exc:
        return RenderResult(
            persona=persona.name,
            index=index,
            intent=turn.intent,
            text=turn.text,
            ok=False,
            duration_ms=int((time.monotonic() - started) * 1000),
            error=str(exc),
        )

    return RenderResult(
        persona=persona.name,
        index=index,
        intent=turn.intent,
        text=turn.text,
        ok=True,
        path=str(path),
        bytes_written=len(audio),
        duration_ms=int((time.monotonic() - started) * 1000),
    )


def render_persona_batch(
    persona_name: str,
    backend: TTSBackend,
    out_dir: Path = DEFAULT_OUTPUT_DIR,
    verbose: bool = True,
) -> List[RenderResult]:
    """Render every turn of one persona's script."""
    persona = get_persona(persona_name)
    script = get_script(persona_name)

    results: List[RenderResult] = []
    for index, turn in enumerate(script.turns, start=1):
        result = render_turn(backend, persona, turn, index, out_dir)
        results.append(result)
        if verbose:
            mark = "ok  " if result.ok else "FAIL"
            size = f"{result.bytes_written:>8,}b" if result.ok else " " * 9
            print(f"  [{mark}] {index:>2}/{len(script)} {size}  {turn.intent:<12} "
                  f"{turn.text[:48]}")
            if not result.ok:
                print(f"         {result.error}")
    return results


def write_results(results: List[RenderResult], out_dir: Path, label: str) -> Path:
    """Persist the batch summary next to the audio it describes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{label}_results.json"
    payload = {
        "label": label,
        "total": len(results),
        "ok": sum(1 for r in results if r.ok),
        "failed": sum(1 for r in results if not r.ok),
        "results": [r.to_dict() for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_backend(args: argparse.Namespace) -> TTSBackend:
    """Pick a backend from CLI flags and environment. Defaults to dry run."""
    if args.dry_run:
        return DryRunBackend()
    return CartesiaTTSBackend(
        api_key=args.api_key or os.environ.get("CARTESIA_API_KEY", ""),
        voice_id=args.voice_id or os.environ.get("CARTESIA_VOICE_ID", ""),
        model_id=args.model_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="persona_tts",
        description="Render candidate script turns to audio.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--persona", choices=sorted(PERSONAS), help="render one persona")
    target.add_argument(
        "--all-personas", action="store_true", help="render every persona"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="make no network calls and write no audio (default when no key is set)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--output", type=Path, help="render only the first turn, to this exact path"
    )
    parser.add_argument("--api-key", help="overrides CARTESIA_API_KEY")
    parser.add_argument("--voice-id", help="overrides CARTESIA_VOICE_ID")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = not args.quiet

    # Refuse to silently do nothing, and refuse to silently spend money.
    if not args.persona and not args.all_personas:
        args.all_personas = True
        args.dry_run = True
        if verbose:
            print("No target given; listing all personas as a dry run.\n")

    if not args.dry_run and not (args.api_key or os.environ.get("CARTESIA_API_KEY")):
        print(
            "No TTS API key found. Re-run with --dry-run to exercise the "
            "pipeline offline, or set CARTESIA_API_KEY to render real audio.",
            file=sys.stderr,
        )
        return 2

    try:
        backend = build_backend(args)
    except TTSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    targets = [args.persona] if args.persona else sorted(PERSONAS)

    # --output renders a single turn to an exact path, for a quick ear check.
    if args.output:
        persona = get_persona(targets[0])
        turn = get_script(persona.name).turns[0]
        try:
            audio = backend.synthesize(turn.text, persona)
        except TTSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(audio)
        print(f"Rendered {len(audio):,} bytes to {args.output}")
        return 0

    all_results: List[RenderResult] = []
    for name in targets:
        if verbose:
            print(f"\n=== {name} ===")
        all_results.extend(
            render_persona_batch(name, backend, args.output_dir, verbose=verbose)
        )

    label = args.persona or "all_personas"
    summary_path = write_results(all_results, args.output_dir, label)

    ok = sum(1 for r in all_results if r.ok)
    failed = len(all_results) - ok
    print(f"\n{ok} rendered, {failed} failed. Summary: {summary_path}")
    if args.dry_run:
        print("(dry run: no audio was produced and no API was called)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
