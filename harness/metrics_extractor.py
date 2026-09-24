"""Week 3: turn captured sessions into numbers that can be compared.

Layer 2 records *what happened and when*. This layer answers *how did it
behave*, and it does so only from timestamps that layer 2 recorded live. It
never reopens the audio, so the metrics can be rewritten and recomputed against
sessions already on disk without re-running a single conversation.

What can honestly be measured from outside the agent
----------------------------------------------------

The harness sits in the room, not inside the agent. That boundary decides what
is measurable and what is merely assertable.

**Measurable.** The gap between the candidate finishing and the agent starting
to speak. That is a real, externally observable number, and it is the one a
candidate actually experiences.

**Not measurable from here.** Which *stage* consumed that gap. Endpointing,
transcription, model inference and speech synthesis all sit inside that single
interval, and no amount of listening from outside separates them. A claim like
"voice-activity detection adds 180 ms" cannot be made from this data. It needs
the agent's own per-stage instrumentation -- LiveKit agents emit exactly that as
``MetricsCollectedEvent`` -- joined to these sessions by room name.

So this module reports ``response_latency_ms`` as the end-to-end figure it is,
and names the stages it contains rather than pretending to separate them. A
harness that overstates what it measured is worse than one that measures less.

Why medians and percentiles rather than means
---------------------------------------------

A benchmark run is a handful of conversations, and response latency is
right-skewed: one slow first token drags a mean far from anything a speaker
would experience. The median says what usually happens, p95 says how bad the
tail gets, and both are reported with ``n`` so a reader can see how much weight
the number carries. A mean over six samples, quoted alone, is a way to be
confidently wrong.

Simulated sessions are refused
------------------------------

A session recorded under the virtual clock carries ``time_scale`` 0.0. Its
timings are exact simulation, not measurement: no network, no provider, no real
model. Those sessions are rejected here rather than filtered by convention,
because the one thing this project cannot afford is a number that looks real and
is not.
"""

from __future__ import annotations

import argparse
import csv
import json

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from persona_spec import PERSONAS
from session_log import (
    ACTOR_AGENT,
    ACTOR_CANDIDATE,
    AGENT_SPEECH_END,
    AGENT_SPEECH_START,
    BARGE_IN,
    CANDIDATE_SPEECH_END,
    CANDIDATE_SPEECH_START,
    STOP_SCRIPT_EXHAUSTED,
    STOP_TURN_CEILING,
    load_session,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPT_DIR = PROJECT_ROOT / "results" / "transcripts"
DEFAULT_METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

HEALTHY_STOPS = {STOP_SCRIPT_EXHAUSTED, STOP_TURN_CEILING}

# What response_latency_ms actually contains. Stated wherever the number is
# reported, so it can never be quoted as if it were one stage.
LATENCY_COMPONENTS = (
    "endpointing + speech-to-text + model inference + speech synthesis + network"
)


class MetricsError(ValueError):
    """Raised when a session cannot honestly be turned into metrics."""


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    """Nearest-rank percentile. Returns None for an empty sample.

    Nearest-rank rather than interpolated: with six samples, interpolation
    invents a value between two observations and reports it to the reader as
    though it were measured. Nearest-rank always returns a number that actually
    happened.
    """
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(pct / 100 * len(ordered)))))
    return float(ordered[rank - 1])


def summarise(values: Sequence[float]) -> Dict[str, Optional[float]]:
    """Distribution summary. ``n`` is included so the reader can weigh it.

    The median goes through :func:`percentile` rather than ``statistics.median``
    so that both it and p95 obey the same rule. ``statistics.median`` averages
    the two middle values of an even sample, which would report a latency that
    no turn actually took -- the exact thing nearest-rank was chosen to avoid.
    Two different conventions in one summary table is worse than either.
    """
    clean = [float(v) for v in values]
    return {
        "n": len(clean),
        "median_ms": round(percentile(clean, 50), 1) if clean else None,
        "p95_ms": round(percentile(clean, 95), 1) if clean else None,
        "min_ms": round(min(clean), 1) if clean else None,
        "max_ms": round(max(clean), 1) if clean else None,
    }


@dataclass
class SessionMetrics:
    """Everything derivable about one session, without reopening its audio."""

    session_id: str
    persona: str
    language: str
    transport: str
    stop_reason: str
    healthy: bool
    duration_ms: int

    # Candidate stops speaking -> agent starts. The end-to-end figure.
    response_latency_ms: List[int] = field(default_factory=list)

    # Candidate starts talking over the agent -> agent falls silent.
    # Directly answers "does this agent yield when interrupted, and how fast".
    barge_in_yield_ms: List[int] = field(default_factory=list)

    agent_utterance_ms: List[int] = field(default_factory=list)
    candidate_utterance_ms: List[int] = field(default_factory=list)

    overlap_ms: int = 0
    intended_barge_ins: int = 0
    confirmed_barge_ins: int = 0
    agent_turns: int = 0
    candidate_turns: int = 0
    dead_air_events: int = 0

    # The detector's own resolution limit, carried alongside the numbers it
    # produced so nobody reads more precision into them than exists.
    detection_lag_ms: List[int] = field(default_factory=list)

    @property
    def overlap_ratio(self) -> float:
        return round(self.overlap_ms / self.duration_ms, 4) if self.duration_ms else 0.0

    @property
    def barge_in_success_rate(self) -> Optional[float]:
        """Share of attempted interruptions that actually overlapped."""
        if not self.intended_barge_ins:
            return None
        return round(self.confirmed_barge_ins / self.intended_barge_ins, 3)

    def to_dict(self) -> Dict:
        out = asdict(self)
        out["overlap_ratio"] = self.overlap_ratio
        out["barge_in_success_rate"] = self.barge_in_success_rate
        out["response_latency"] = summarise(self.response_latency_ms)
        out["barge_in_yield"] = summarise(self.barge_in_yield_ms)
        out["agent_utterance"] = summarise(self.agent_utterance_ms)
        out["latency_contains"] = LATENCY_COMPONENTS
        return out

    def row(self) -> Dict:
        """Flat one-line view, for the CSV and the console table."""
        latency = summarise(self.response_latency_ms)
        yield_ = summarise(self.barge_in_yield_ms)
        return {
            "session_id": self.session_id,
            "persona": self.persona,
            "language": self.language,
            "transport": self.transport,
            "stop_reason": self.stop_reason,
            "healthy": self.healthy,
            "duration_s": round(self.duration_ms / 1000, 1),
            "agent_turns": self.agent_turns,
            "candidate_turns": self.candidate_turns,
            "latency_n": latency["n"],
            "latency_median_ms": latency["median_ms"],
            "latency_p95_ms": latency["p95_ms"],
            "yield_median_ms": yield_["median_ms"],
            "overlap_ratio": self.overlap_ratio,
            "barge_in_success_rate": self.barge_in_success_rate,
        }


def _language_of(persona: str) -> str:
    spec = PERSONAS.get(persona)
    return spec.language if spec else "unknown"


def extract_session_metrics(
    session: Path | Dict, allow_simulated: bool = False
) -> SessionMetrics:
    """Derive metrics for one captured session.

    Accepts a path or an already-loaded record, so callers that have just run a
    session need not round-trip it through disk.
    """
    data = load_session(session) if isinstance(session, Path) else session

    time_scale = data.get("time_scale", 1.0)
    if time_scale != 1.0 and not allow_simulated:
        raise MetricsError(
            f"session {data.get('session_id')} was recorded with time_scale "
            f"{time_scale}, meaning simulated time against a simulated agent. "
            "Its timings are exact but they measure nothing real. Re-run "
            "against a live agent, or pass allow_simulated=True to inspect the "
            "shape of the output knowing the numbers are not measurements."
        )

    persona = data.get("persona", "unknown")
    metrics = SessionMetrics(
        session_id=data.get("session_id", "?"),
        persona=persona,
        language=_language_of(persona),
        transport=data.get("transport") or "unknown",
        stop_reason=data.get("stop_reason") or "unknown",
        healthy=data.get("stop_reason") in HEALTHY_STOPS,
        duration_ms=int(data.get("duration_ms", 0)),
    )

    events = sorted(data.get("events", []), key=lambda e: e["t_ms"])

    # --- response latency ---------------------------------------------------
    # Pair each candidate utterance end with the next agent utterance start.
    # Anything in between that is not an agent start (a second candidate turn,
    # a dead-air strike) means the agent never answered that turn, and pairing
    # across it would silently invent a latency spanning unrelated events.
    pending_end: Optional[int] = None
    for event in events:
        kind = event["kind"]
        if kind == CANDIDATE_SPEECH_END:
            pending_end = event["t_ms"]
        elif kind == AGENT_SPEECH_START:
            if pending_end is not None:
                metrics.response_latency_ms.append(event["t_ms"] - pending_end)
                pending_end = None
        elif kind == CANDIDATE_SPEECH_START and pending_end is not None:
            # The candidate spoke again before any reply: no latency to record.
            pending_end = None

    # --- barge-in yield -----------------------------------------------------
    # From the moment the candidate talks over the agent to the agent actually
    # falling silent. Only counted when an agent utterance was genuinely open.
    open_agent_turn = False
    barge_at: Optional[int] = None
    for event in events:
        kind = event["kind"]
        if kind == AGENT_SPEECH_START:
            open_agent_turn = True
        elif kind == AGENT_SPEECH_END:
            if barge_at is not None and open_agent_turn:
                metrics.barge_in_yield_ms.append(event["t_ms"] - barge_at)
            barge_at = None
            open_agent_turn = False
        elif kind == BARGE_IN and open_agent_turn and barge_at is None:
            barge_at = event["t_ms"]

    # --- detector resolution, carried alongside -----------------------------
    for event in events:
        lag = event.get("data", {}).get("detection_lag_ms")
        if isinstance(lag, int):
            metrics.detection_lag_ms.append(lag)

    metrics.dead_air_events = sum(1 for e in events if e["kind"] == "dead_air")

    # --- per-turn figures ---------------------------------------------------
    for turn in data.get("turns", []):
        duration = turn.get("duration_ms")
        if turn["actor"] == ACTOR_AGENT:
            metrics.agent_turns += 1
            if duration is not None:
                metrics.agent_utterance_ms.append(duration)
        elif turn["actor"] == ACTOR_CANDIDATE:
            metrics.candidate_turns += 1
            if duration is not None:
                metrics.candidate_utterance_ms.append(duration)
            metrics.overlap_ms += turn.get("overlap_ms") or 0
            if turn.get("intended_barge_in"):
                metrics.intended_barge_ins += 1
            if turn.get("barge_in"):
                metrics.confirmed_barge_ins += 1

    return metrics


def aggregate(records: Iterable[SessionMetrics], by: str = "language") -> Dict:
    """Group sessions and summarise each group.

    Grouping by language is the comparison the project exists to make: a single
    global endpointing threshold is the norm in agent implementations, and a
    threshold tuned against one language is not automatically right for the
    others that agent serves. Grouping by persona answers a different question --
    which kind of speaker this agent handles worst.
    """
    if by not in ("language", "persona", "transport"):
        raise MetricsError(f"cannot group by {by!r}; use language, persona or transport")

    groups: Dict[str, List[SessionMetrics]] = {}
    for record in records:
        groups.setdefault(getattr(record, by), []).append(record)

    out: Dict[str, Dict] = {}
    for key, items in sorted(groups.items()):
        latencies = [v for m in items for v in m.response_latency_ms]
        yields = [v for m in items for v in m.barge_in_yield_ms]
        lags = [v for m in items for v in m.detection_lag_ms]
        intended = sum(m.intended_barge_ins for m in items)
        confirmed = sum(m.confirmed_barge_ins for m in items)
        out[key] = {
            "sessions": len(items),
            "healthy_sessions": sum(1 for m in items if m.healthy),
            "response_latency": summarise(latencies),
            "barge_in_yield": summarise(yields),
            "overlap_ratio_median": (
                round(percentile([m.overlap_ratio for m in items], 50), 4)
                if items
                else None
            ),
            "intended_barge_ins": intended,
            "confirmed_barge_ins": confirmed,
            "barge_in_success_rate": (
                round(confirmed / intended, 3) if intended else None
            ),
            # The floor under every latency above. Quoting a difference smaller
            # than this would be quoting detector resolution, not behaviour.
            "detection_lag": summarise(lags),
            "latency_contains": LATENCY_COMPONENTS,
        }
    return out


def compare(aggregated: Dict, baseline: str) -> Dict:
    """Difference each group's median latency against a baseline group.

    This is the shape of the project's headline claim -- one language slower
    than another -- so the comparison is computed rather than eyeballed, and the
    detector's resolution is carried with it. A difference at or below that
    floor is not evidence of anything.
    """
    if baseline not in aggregated:
        raise MetricsError(
            f"baseline {baseline!r} not present; have: {', '.join(sorted(aggregated))}"
        )
    base = aggregated[baseline]["response_latency"]["median_ms"]
    if base is None:
        raise MetricsError(f"baseline {baseline!r} has no latency samples")

    floor = aggregated[baseline]["detection_lag"]["p95_ms"] or 0.0
    out = {}
    for key, group in aggregated.items():
        median = group["response_latency"]["median_ms"]
        if median is None:
            continue
        delta = round(median - base, 1)
        out[key] = {
            "median_ms": median,
            "delta_vs_baseline_ms": delta,
            "n": group["response_latency"]["n"],
            "above_detector_floor": abs(delta) > floor,
            "detector_floor_ms": floor,
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_sessions(
    directory: Path, allow_simulated: bool = False
) -> tuple[List[SessionMetrics], List[str]]:
    """Read every session in a directory, reporting what was skipped and why."""
    records, skipped = [], []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(extract_session_metrics(path, allow_simulated))
        except MetricsError as exc:
            skipped.append(f"{path.name}: {exc}")
        except (KeyError, json.JSONDecodeError) as exc:
            skipped.append(f"{path.name}: unreadable ({exc})")
    return records, skipped


def write_outputs(
    records: List[SessionMetrics], aggregated: Dict, out_dir: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "metrics.json"
    json_path.write_text(
        json.dumps(
            {
                "sessions": [m.to_dict() for m in records],
                "by_language": aggregated,
                "latency_contains": LATENCY_COMPONENTS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = out_dir / "sessions.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].row()))
            writer.writeheader()
            for record in records:
                writer.writerow(record.row())
    return json_path, csv_path


def print_table(records: List[SessionMetrics]) -> None:
    if not records:
        print("  (no sessions)")
        return
    header = (
        f"  {'PERSONA':<20}{'LANG':<6}{'STOP':<18}{'TURNS':>6}"
        f"{'LAT n':>7}{'LAT med':>9}{'LAT p95':>9}{'YIELD':>8}{'OVLP':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for m in records:
        row = m.row()
        print(
            f"  {row['persona']:<20}{row['language']:<6}{row['stop_reason']:<18}"
            f"{row['agent_turns']:>6}"
            f"{row['latency_n']:>7}"
            f"{_fmt(row['latency_median_ms']):>9}"
            f"{_fmt(row['latency_p95_ms']):>9}"
            f"{_fmt(row['yield_median_ms']):>8}"
            f"{row['overlap_ratio']:>7.3f}"
        )


def _fmt(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.0f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metrics_extractor",
        description="Turn captured sessions into comparable numbers.",
    )
    parser.add_argument("--dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument(
        "--by", choices=("language", "persona", "transport"), default="language"
    )
    parser.add_argument(
        "--baseline",
        help="group to difference the others against, e.g. 'en'",
    )
    parser.add_argument(
        "--allow-simulated",
        action="store_true",
        help=(
            "include sessions recorded under the virtual clock. Their timings "
            "are simulation, not measurement -- for inspecting output shape only"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.dir.exists():
        print(f"error: {args.dir} does not exist", file=sys.stderr)
        return 2

    records, skipped = load_sessions(args.dir, args.allow_simulated)

    if skipped:
        print(f"Skipped {len(skipped)} session(s):")
        for reason in skipped:
            print(f"  - {reason}")
        print()

    if not records:
        print(
            "No measurable sessions found.\n"
            "  Sessions recorded with --clock virtual are simulation and are "
            "refused by default.\n"
            "  Run against a live agent (--transport livekit), or pass "
            "--allow-simulated to\n"
            "  inspect the output shape knowing the numbers mean nothing.",
            file=sys.stderr,
        )
        return 1

    print(f"{len(records)} session(s)\n")
    print_table(records)

    aggregated = aggregate(records, by=args.by)
    print(f"\nBy {args.by}:")
    for key, group in aggregated.items():
        lat = group["response_latency"]
        print(
            f"  {key:<12} sessions={group['sessions']:<3} "
            f"latency n={lat['n']:<3} median={_fmt(lat['median_ms'])}ms "
            f"p95={_fmt(lat['p95_ms'])}ms  "
            f"barge-in {group['confirmed_barge_ins']}/{group['intended_barge_ins']}"
        )

    if args.baseline:
        print(f"\nAgainst baseline {args.baseline!r}:")
        for key, cmp in compare(aggregated, args.baseline).items():
            if key == args.baseline:
                continue
            verdict = (
                "above detector resolution"
                if cmp["above_detector_floor"]
                else "WITHIN detector resolution -- not evidence"
            )
            print(
                f"  {key:<12} {cmp['delta_vs_baseline_ms']:+.0f}ms "
                f"(n={cmp['n']})  {verdict}"
            )

    json_path, csv_path = write_outputs(records, aggregated, args.out_dir)
    print(f"\nlatency = {LATENCY_COMPONENTS}")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
