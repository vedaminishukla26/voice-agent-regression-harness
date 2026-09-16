"""Timestamped record of one audio session.

This is the artifact week 2 exists to produce. Layers above it do not re-open
the audio; they read this file. So the rule it enforces is that **every
timestamp is recorded at the moment the thing happened**, never reconstructed
afterwards from durations and assumptions.

That distinction is not pedantry. Barge-in is defined entirely by two events
overlapping in time, and an overlap cannot be recovered from a transcript, from
turn durations, or from anything else that has already collapsed the timeline.
If the loop does not capture it live, it is gone.

All offsets are integer milliseconds from session start, taken from a monotonic
clock. Monotonic because a wall clock can step backwards mid-session -- NTP
correction, a laptop resuming from sleep -- and a negative latency in a results
file is the kind of thing nobody notices until it has been quoted in a report.
The wall-clock time of session start is recorded once, separately, so a session
can still be located in a log elsewhere.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

# --- event kinds -----------------------------------------------------------
# Deliberately flat strings rather than an enum: they are written to JSON and
# read by other tools, so the wire format is the interface.

SESSION_START = "session_start"
SESSION_END = "session_end"
AGENT_JOINED = "agent_joined"
AGENT_LEFT = "agent_left"
AGENT_SPEECH_START = "agent_speech_start"
AGENT_SPEECH_END = "agent_speech_end"
CANDIDATE_SPEECH_START = "candidate_speech_start"
CANDIDATE_SPEECH_END = "candidate_speech_end"
AGENT_TRANSCRIPT = "agent_transcript"
TURN_SELECTED = "turn_selected"
BARGE_IN = "barge_in"
DEAD_AIR = "dead_air"
WARNING = "warning"

ACTOR_AGENT = "agent"
ACTOR_CANDIDATE = "candidate"
ACTOR_HARNESS = "harness"

# --- stop reasons ----------------------------------------------------------
# Why a session ended is the first thing to look at when its numbers are odd, so
# it is a required field rather than something inferred from event order.

STOP_SCRIPT_EXHAUSTED = "script_exhausted"
STOP_TURN_CEILING = "turn_ceiling"
STOP_WALL_CLOCK = "wall_clock_timeout"
STOP_AGENT_NEVER_JOINED = "agent_never_joined"
STOP_AGENT_LEFT = "agent_left"
STOP_AGENT_SILENT = "agent_silent"
STOP_ERROR = "error"


class Clock(Protocol):
    """Monotonic millisecond source. Injected so tests need no real time."""

    def now_ms(self) -> int: ...


class MonotonicClock:
    """Milliseconds since construction, from ``time.monotonic``."""

    def __init__(self) -> None:
        self._origin = time.monotonic()

    def now_ms(self) -> int:
        return int((time.monotonic() - self._origin) * 1000)


class FakeClock:
    """Manually advanced clock, so the loop's timing logic is testable.

    The turn-taking state machine is mostly a set of decisions about elapsed
    time. Testing it against a real clock would mean sleeping through every
    case, which makes the suite slow enough that it stops being run -- the
    failure mode INFRA.md is written to avoid.
    """

    def __init__(self, start_ms: int = 0) -> None:
        self._now = start_ms

    def now_ms(self) -> int:
        return self._now

    def advance(self, ms: int) -> int:
        if ms < 0:
            raise ValueError("cannot advance a clock backwards")
        self._now += ms
        return self._now

    def set(self, ms: int) -> int:
        self._now = ms
        return self._now


class Scheduler(Protocol):
    """A clock that can also be waited on.

    The turn-taking loop is almost entirely decisions about elapsed time, so the
    thing to make injectable is not just *what time is it* but *wait this long*.
    With both behind one object, a session can be run against compressed time
    without the loop knowing.
    """

    def now_ms(self) -> int: ...

    async def sleep_ms(self, ms: int) -> None: ...

    @property
    def time_scale(self) -> float: ...


class RealScheduler(MonotonicClock):
    """Wall-clock scheduling. The only scheduler valid for real measurement."""

    time_scale = 1.0

    async def sleep_ms(self, ms: int) -> None:
        if ms > 0:
            await asyncio.sleep(ms / 1000)


class VirtualScheduler:
    """Simulated time that advances only when every task is waiting on it.

    A realistic session is tens of seconds of mostly waiting. Running the test
    suite through that in real time would take minutes, and a suite that takes
    minutes stops being run -- the exact rot INFRA.md is written against.

    The obvious shortcut, dividing every sleep by a speed factor, does not
    survive contact with a real event loop. Timer resolution is around 15 ms on
    Windows, so a sleep asking for 1 ms takes fifteen, and under a 20x factor
    that single frame appears to consume 300 ms of session time. The detector
    then sees frames spaced far wider than the thresholds counted in frames, and
    the state machine under test is no longer the one that runs in production.
    Compressed real time does not just distort the numbers; it changes the
    behaviour, which makes it useless for testing as well as for measuring.

    So time here is not compressed, it is simulated. ``sleep_ms`` registers a
    wake-up and blocks. A driver advances the clock straight to the earliest
    pending wake-up once nothing else can make progress. Sleeps are therefore
    exact -- a 20 ms frame is 20 ms, always -- the session is deterministic, and
    a three-minute conversation completes in well under a second.

    **Measurements from a virtual session are still not real measurements.**
    Nothing here models network latency or provider response time; the agent is
    simulated. ``time_scale`` is recorded as 0.0 so that anything downstream can
    refuse such a session outright rather than relying on the operator to
    remember which runs were real.
    """

    time_scale = 0.0

    def __init__(self) -> None:
        self._now = 0
        self._waiters: List[tuple] = []
        self._sequence = 0
        self._driver: Optional[asyncio.Task] = None

    def now_ms(self) -> int:
        return self._now

    async def sleep_ms(self, ms: int) -> None:
        if ms <= 0:
            await asyncio.sleep(0)
            return
        event = asyncio.Event()
        self._sequence += 1
        heapq.heappush(self._waiters, (self._now + ms, self._sequence, event))
        await event.wait()

    def _advance(self) -> bool:
        """Jump to the earliest pending wake-up. False if nothing is waiting."""
        if not self._waiters:
            return False
        self._now = max(self._now, self._waiters[0][0])
        while self._waiters and self._waiters[0][0] <= self._now:
            heapq.heappop(self._waiters)[2].set()
        return True

    async def _drive(self) -> None:
        # Yield repeatedly before advancing so every task that can still run
        # gets to, and registers its next sleep, before the clock moves. Without
        # this the clock could step past a wake-up that had not been booked yet.
        while True:
            for _ in range(8):
                await asyncio.sleep(0)
            if not self._advance():
                await asyncio.sleep(0)

    def start(self) -> None:
        if self._driver is None:
            self._driver = asyncio.create_task(self._drive(), name="virtual-clock")

    async def stop(self) -> None:
        if self._driver is None:
            return
        self._driver.cancel()
        try:
            await self._driver
        except asyncio.CancelledError:
            pass
        self._driver = None
        # Release anything still asleep so shutdown cannot hang on a wake-up
        # that will now never arrive.
        while self._waiters:
            heapq.heappop(self._waiters)[2].set()


async def race_timeout(scheduler: Scheduler, awaitable, timeout_ms: int) -> bool:
    """Await something, giving up after ``timeout_ms`` of *scheduler* time.

    ``asyncio.wait_for`` measures real seconds, which is correct under a real
    clock and meaningless under a simulated one. Every timeout in the loop goes
    through here instead, so the same code is correct under both and the tests
    exercise the production path rather than a parallel one.

    Returns True if the awaitable finished, False if the timeout won.
    """
    target = asyncio.ensure_future(awaitable)
    timer = asyncio.ensure_future(scheduler.sleep_ms(timeout_ms))
    try:
        done, _ = await asyncio.wait(
            {target, timer}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        if not timer.done():
            timer.cancel()
    if target in done:
        await target
        return True
    target.cancel()
    try:
        await target
    except asyncio.CancelledError:
        pass
    return False


@dataclass(frozen=True)
class SessionEvent:
    """One thing that happened, at a known offset from session start."""

    t_ms: int
    kind: str
    actor: str
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.t_ms < 0:
            raise ValueError(f"event {self.kind!r}: t_ms must not be negative")
        if not self.kind:
            raise ValueError("event kind must not be empty")
        if self.actor not in (ACTOR_AGENT, ACTOR_CANDIDATE, ACTOR_HARNESS):
            raise ValueError(f"event {self.kind!r}: unknown actor {self.actor!r}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TurnRecord:
    """One utterance by one party, as a convenience view over the events.

    Two separate facts about interruption are kept, because conflating them
    produces a barge-in metric that cannot be trusted:

    ``intended_barge_in``
        The harness decided to talk over the agent. This is an *action*, known
        at the time it is taken.

    ``overlap_ms`` and ``barge_in``
        Whether the two utterances genuinely overlapped, and by how much. This
        is an *outcome*. It cannot be known when the turn starts, because the
        detector cannot declare the agent's utterance finished until it has
        heard enough silence to be sure -- by which point the harness has
        already committed. It is filled in once the session ends and every
        agent segment is closed.

    An intended barge-in that produced no overlap is not a barge-in. It is the
    harness having aimed at a moving target and missed, and it must not be
    counted as the thing it was trying to be.
    """

    index: int
    actor: str
    start_ms: int
    end_ms: Optional[int] = None
    intent: Optional[str] = None
    text: Optional[str] = None
    intended_barge_in: bool = False
    barge_in: bool = False
    overlap_ms: Optional[int] = None

    @property
    def duration_ms(self) -> Optional[int]:
        return None if self.end_ms is None else self.end_ms - self.start_ms

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["duration_ms"] = self.duration_ms
        return out


class SessionLog:
    """Accumulates events for one session and writes them out.

    Kept append-only and free of any analysis. Deriving metrics here would mean
    the metrics could only ever be recomputed by re-running the session; keeping
    them out means week 3 can be rewritten against sessions already on disk.
    """

    def __init__(
        self,
        persona: str,
        room: str,
        clock: Optional[Clock] = None,
        session_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.persona = persona
        self.room = room
        self.clock = clock or MonotonicClock()
        self.config = config or {}
        self.started_at_utc = datetime.now(timezone.utc).isoformat()
        self.events: List[SessionEvent] = []
        self.turns: List[TurnRecord] = []
        self.stop_reason: Optional[str] = None
        self.error: Optional[str] = None
        self.transport: Optional[str] = None

        self.record(SESSION_START, ACTOR_HARNESS, persona=persona, room=room)

    # -- recording ----------------------------------------------------------

    def record(
        self, kind: str, actor: str, t_ms: Optional[int] = None, **data: Any
    ) -> SessionEvent:
        """Append an event. ``t_ms`` overrides the clock for retroactive stamps.

        The override exists for the speech detector, which knows an utterance
        began earlier than the moment it could prove it. Passing the true
        timestamp through is the whole reason the detector bothers to compute
        one. Events are kept sorted so a retroactive stamp does not leave the
        log out of order for anything that reads it positionally.
        """
        event = SessionEvent(
            t_ms=self.clock.now_ms() if t_ms is None else t_ms,
            kind=kind,
            actor=actor,
            data=data,
        )
        self.events.append(event)
        if len(self.events) > 1 and event.t_ms < self.events[-2].t_ms:
            self.events.sort(key=lambda e: e.t_ms)
        return event

    def warn(self, message: str, **data: Any) -> SessionEvent:
        """Record something that went wrong but did not stop the session.

        Surfacing these in the record rather than only on stderr means a session
        that produced odd numbers carries its own explanation.
        """
        return self.record(WARNING, ACTOR_HARNESS, message=message, **data)

    def begin_turn(
        self,
        actor: str,
        start_ms: int,
        intent: Optional[str] = None,
        text: Optional[str] = None,
        intended_barge_in: bool = False,
    ) -> TurnRecord:
        turn = TurnRecord(
            index=len(self.turns) + 1,
            actor=actor,
            start_ms=start_ms,
            intent=intent,
            text=text,
            intended_barge_in=intended_barge_in,
        )
        self.turns.append(turn)
        return turn

    def resolve_overlaps(self) -> None:
        """Fill in true speech overlap once every segment has been closed.

        Runs at session end, over timestamps that were all recorded live. The
        overlap is *derived* from recorded facts, not reconstructed from
        durations and assumptions -- the distinction :mod:`audio_loop` depends
        on. It is computed here, once, so that the session record answers the
        question on its own rather than leaving every later reader to redo it
        and possibly disagree.
        """
        agent_spans = [
            (t.start_ms, t.end_ms)
            for t in self.turns
            if t.actor == ACTOR_AGENT and t.end_ms is not None
        ]
        for turn in self.turns:
            if turn.actor != ACTOR_CANDIDATE or turn.end_ms is None:
                continue
            overlap = 0
            for start, end in agent_spans:
                overlap += max(0, min(turn.end_ms, end) - max(turn.start_ms, start))
            turn.overlap_ms = overlap
            turn.barge_in = overlap > 0

    def end_turn(self, actor: str, end_ms: int) -> Optional[TurnRecord]:
        """Close the most recent open turn for ``actor``."""
        for turn in reversed(self.turns):
            if turn.actor == actor and turn.end_ms is None:
                turn.end_ms = end_ms
                return turn
        return None

    def open_turn(self, actor: str) -> Optional[TurnRecord]:
        for turn in reversed(self.turns):
            if turn.actor == actor and turn.end_ms is None:
                return turn
        return None

    def finish(self, stop_reason: str, error: Optional[str] = None) -> None:
        if self.stop_reason is not None:
            return
        self.stop_reason = stop_reason
        self.error = error
        self.record(SESSION_END, ACTOR_HARNESS, stop_reason=stop_reason, error=error)

    # -- views --------------------------------------------------------------

    @property
    def duration_ms(self) -> int:
        return self.events[-1].t_ms if self.events else 0

    def events_of(self, *kinds: str) -> List[SessionEvent]:
        wanted = set(kinds)
        return [e for e in self.events if e.kind in wanted]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "persona": self.persona,
            "room": self.room,
            "transport": self.transport,
            "started_at_utc": self.started_at_utc,
            "duration_ms": self.duration_ms,
            # 1.0 means real time. Anything else means the session was run under
            # compressed time and its timings are control-flow evidence only,
            # never measurement. Recorded so downstream layers can enforce that
            # rather than trusting the operator to remember.
            "time_scale": getattr(self.clock, "time_scale", 1.0),
            "stop_reason": self.stop_reason,
            "error": self.error,
            "config": self.config,
            "counts": {
                "events": len(self.events),
                "turns": len(self.turns),
                "candidate_turns": sum(
                    1 for t in self.turns if t.actor == ACTOR_CANDIDATE
                ),
                "agent_turns": sum(1 for t in self.turns if t.actor == ACTOR_AGENT),
                # Intended is what the harness tried to do; confirmed is what
                # actually overlapped. They differ whenever the agent stopped
                # talking between the decision and the first frame going out,
                # and reporting only one of them would hide that.
                "intended_barge_ins": sum(1 for t in self.turns if t.intended_barge_in),
                "barge_ins": sum(1 for t in self.turns if t.barge_in),
                "overlap_ms": sum(t.overlap_ms or 0 for t in self.turns),
                "warnings": len(self.events_of(WARNING)),
            },
            "turns": [t.to_dict() for t in self.turns],
            "events": [e.to_dict() for e in self.events],
        }

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.session_id}_{self.persona}.json"
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def summary_line(self) -> str:
        counts = self.to_dict()["counts"]
        return (
            f"{self.session_id} {self.persona:<20} "
            f"{self.duration_ms / 1000:>6.1f}s  "
            f"candidate={counts['candidate_turns']:<3} "
            f"agent={counts['agent_turns']:<3} "
            f"barge_in={counts['barge_ins']:<3} "
            f"stop={self.stop_reason}"
        )


def load_session(path: Path) -> Dict[str, Any]:
    """Read a session record back. The entry point for week 3."""
    return json.loads(path.read_text(encoding="utf-8"))
