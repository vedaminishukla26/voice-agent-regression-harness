"""Week 2: close the audio loop.

Publishes rendered persona audio into a real room, lets the agent under test
respond over its own pipeline, and captures both sides with timestamps.

This is the layer that makes the harness worth building. Everything in week 1
could be done against text; nothing here can. Publishing real audio is what
exercises speech-to-text, voice-activity detection, endpointing, barge-in and
text-to-speech -- the components where the interesting regressions actually live.

Three commitments hold this module together.

**The agent under test is a separate process.** The harness joins a room and
behaves like a candidate. It does not import, start or configure the agent, and
it holds none of the agent's prompt or model settings. Anything proprietary
therefore stays where it already lives, in whatever runs the agent, and this
repository stays runnable against any agent on the platform.

**Every limit is a wall clock, not a conversational judgement.** Two systems
that each wait for the other to speak will wait forever, and two systems that
each react to the other can feed back indefinitely. Neither is prevented by
careful turn-taking logic, because both arise when the turn-taking logic is
already confused. So the session is bounded by elapsed time, by a turn ceiling,
and by an agent-silence timeout, each enforced regardless of what the
conversation believes is happening.

**Timing is recorded, never reconstructed.** Barge-in only exists as an overlap
between two live events. Reconstructing it afterwards from durations is not
merely less accurate, it is impossible. See :mod:`session_log`.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from audio_io import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    PcmClip,
    StreamingWavWriter,
    concat,
    estimate_duration_ms,
    synthesize_speech_like,
)
from candidate_scripts import CandidateScript, CandidateTurn, get_script
from persona_spec import PERSONAS, PersonaSpec, get_persona
from session_log import (
    ACTOR_AGENT,
    ACTOR_CANDIDATE,
    ACTOR_HARNESS,
    AGENT_JOINED,
    AGENT_LEFT,
    AGENT_SPEECH_END,
    AGENT_SPEECH_START,
    AGENT_TRANSCRIPT,
    BARGE_IN,
    CANDIDATE_SPEECH_END,
    CANDIDATE_SPEECH_START,
    DEAD_AIR,
    STOP_AGENT_LEFT,
    STOP_AGENT_NEVER_JOINED,
    STOP_AGENT_SILENT,
    STOP_ERROR,
    STOP_SCRIPT_EXHAUSTED,
    STOP_TURN_CEILING,
    STOP_WALL_CLOCK,
    TURN_SELECTED,
    RealScheduler,
    Scheduler,
    SessionLog,
    VirtualScheduler,
    race_timeout,
)
from transport import (
    END_OF_AUDIO,
    AgentFrame,
    LiveKitConfig,
    LiveKitTransport,
    LoopbackAgent,
    LoopbackTransport,
    Transport,
    TransportError,
)
from vad import SPEECH_END, SPEECH_START, SpeechDetector, VadConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRANSCRIPT_DIR = PROJECT_ROOT / "results" / "transcripts"
DEFAULT_AUDIO_DIR = PROJECT_ROOT / "results" / "audio"


@dataclass
class SessionConfig:
    """Everything that bounds and shapes one session."""

    persona: str
    room: str = "harness"

    # --- hard limits, enforced independently of conversational state ---------
    max_turns: int = 12
    max_wall_clock_ms: int = 180_000
    agent_join_timeout_ms: int = 20_000

    # How long to wait for the agent to say anything before calling it silent.
    # Generous: a real agent doing retrieval or a slow first token can sit quiet
    # for several seconds without being broken.
    agent_silence_timeout_ms: int = 15_000

    # Consecutive silent waits tolerated before the session is abandoned.
    max_dead_air_strikes: int = 2

    # --- behaviour -----------------------------------------------------------
    seed: int = 0
    frame_ms: int = DEFAULT_FRAME_MS
    sample_rate: int = DEFAULT_SAMPLE_RATE
    vad: VadConfig = field(default_factory=VadConfig)

    # Window after the agent starts speaking in which a barging persona cuts in.
    barge_in_after_ms: int = 700

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.max_wall_clock_ms <= 0:
            raise ValueError("max_wall_clock_ms must be positive")
        if self.vad.frame_ms != self.frame_ms:
            raise ValueError(
                f"vad.frame_ms ({self.vad.frame_ms}) must match session frame_ms "
                f"({self.frame_ms}); the detector's thresholds are counted in frames"
            )

    def to_dict(self) -> Dict:
        return {
            "persona": self.persona,
            "room": self.room,
            "max_turns": self.max_turns,
            "max_wall_clock_ms": self.max_wall_clock_ms,
            "agent_join_timeout_ms": self.agent_join_timeout_ms,
            "agent_silence_timeout_ms": self.agent_silence_timeout_ms,
            "max_dead_air_strikes": self.max_dead_air_strikes,
            "seed": self.seed,
            "frame_ms": self.frame_ms,
            "sample_rate": self.sample_rate,
            "barge_in_after_ms": self.barge_in_after_ms,
            "vad": {
                "threshold_rms": self.vad.threshold_rms,
                "start_ms": self.vad.start_ms,
                "end_ms": self.vad.end_ms,
            },
        }


class TurnSelector:
    """Chooses which scripted turn the candidate says next.

    A script is a pool of in-character responses, not a fixed dialogue, so
    something has to decide the order. The default is script order, because a
    reproducible benchmark wants the same utterances in the same sequence every
    run unless there is a reason to deviate.

    There is one deviation, and it earns its place. If the agent asks the *same
    question twice*, the selector answers the second time with a turn of a
    different intent than the one that drew the repeat. That makes the most
    valuable behaviour in the whole harness observable: an agent that correctly
    re-asks a dodged question produces a visibly different conversation from one
    that silently moves on, and the difference shows up in the transcript rather
    than having to be inferred.
    """

    def __init__(self, script: CandidateScript, rng: random.Random) -> None:
        self.script = script
        self.rng = rng
        self.used: List[int] = []
        self._agent_lines: List[str] = []

    @staticmethod
    def _normalise(text: str) -> str:
        return " ".join(text.lower().split()).rstrip("?.!,")

    def note_agent_line(self, text: str) -> bool:
        """Record an agent utterance. Returns True if it repeats an earlier one."""
        key = self._normalise(text)
        repeated = key in [self._normalise(t) for t in self._agent_lines]
        self._agent_lines.append(text)
        return repeated

    def _remaining(self) -> List[int]:
        return [i for i in range(len(self.script.turns)) if i not in self.used]

    def next_turn(self, agent_repeated: bool = False) -> Optional[CandidateTurn]:
        remaining = self._remaining()
        if not remaining:
            return None

        index = remaining[0]
        if agent_repeated and self.used:
            last_intent = self.script.turns[self.used[-1]].intent
            differing = [
                i for i in remaining if self.script.turns[i].intent != last_intent
            ]
            if differing:
                index = differing[0]

        self.used.append(index)
        return self.script.turns[index]


class ClipSource:
    """Supplies audio for a turn, caching so a run costs at most one render.

    Real synthesis costs money per call and varies slightly between calls. Both
    are bad in a regression harness: the first makes repeated runs expensive,
    the second makes two runs of the same benchmark not strictly comparable.
    Caching to disk fixes both -- the audio for a given persona and turn is
    rendered once and then reused verbatim for every future run.
    """

    def __init__(
        self,
        cache_dir: Path,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        tts=None,
        seed: int = 0,
    ) -> None:
        self.cache_dir = cache_dir
        self.sample_rate = sample_rate
        self.tts = tts
        self.seed = seed
        self.rendered = 0
        self.cache_hits = 0
        self.synthetic = 0

    def _cache_path(self, persona: PersonaSpec, index: int) -> Path:
        kind = "tts" if self.tts is not None else "synth"
        return self.cache_dir / f"{persona.name}_{index:02d}_{kind}_{self.sample_rate}.wav"

    def clip_for(
        self, persona: PersonaSpec, turn: CandidateTurn, index: int
    ) -> PcmClip:
        path = self._cache_path(persona, index)
        if path.exists():
            self.cache_hits += 1
            return PcmClip.from_wav(path)

        if self.tts is not None:
            pcm = self.tts.synthesize_pcm(turn.text, persona, self.sample_rate)
            clip = PcmClip(pcm, sample_rate=self.sample_rate)
            self.rendered += 1
        else:
            # No key configured. Audio of the right *length* still exercises
            # every timing path; only intelligibility is lost, and nothing in
            # week 2 reads the words.
            clip = synthesize_speech_like(
                estimate_duration_ms(turn.text, persona.speech_rate),
                seed=self.seed + index,
                sample_rate=self.sample_rate,
            )
            self.synthetic += 1

        clip.to_wav(path)
        return clip

    def to_dict(self) -> Dict:
        return {
            "rendered": self.rendered,
            "cache_hits": self.cache_hits,
            "synthetic": self.synthetic,
            "cache_dir": str(self.cache_dir),
        }


class AudioSession:
    """One full-duplex conversation against a live agent."""

    def __init__(
        self,
        config: SessionConfig,
        transport: Transport,
        scheduler: Scheduler,
        clip_source: ClipSource,
        transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
        audio_dir: Path = DEFAULT_AUDIO_DIR,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.transport = transport
        self.scheduler = scheduler
        self.clip_source = clip_source
        self.transcript_dir = transcript_dir
        self.audio_dir = audio_dir
        self.verbose = verbose

        self.persona = get_persona(config.persona)
        self.script = get_script(config.persona)
        self.rng = random.Random(config.seed)
        self.selector = TurnSelector(self.script, self.rng)
        self.detector = SpeechDetector(config.vad)

        self.log = SessionLog(
            persona=config.persona,
            room=config.room,
            clock=scheduler,
            config=config.to_dict(),
        )
        self.log.transport = transport.name

        self._agent_speaking = False
        self._agent_speech_started_ms: Optional[int] = None
        self._last_barged_start_ms: Optional[int] = None
        self._transition = asyncio.Event()
        self._agent_gone = asyncio.Event()
        self._published: List[PcmClip] = []
        self._agent_writer: Optional[StreamingWavWriter] = None

    # -- time helpers --------------------------------------------------------

    def _elapsed_ms(self) -> int:
        return self.scheduler.now_ms()

    def _budget_left_ms(self) -> int:
        return self.config.max_wall_clock_ms - self._elapsed_ms()

    # -- agent audio consumption --------------------------------------------

    async def _consume_agent_audio(self) -> None:
        """Feed arriving frames through the detector and record what it finds.

        Runs for the whole session as its own task. Everything else in the
        session reacts to the events this produces.
        """
        while True:
            frame: Optional[AgentFrame] = await self.transport.agent_audio.get()
            if frame is END_OF_AUDIO:
                for event in self.detector.flush(self._elapsed_ms()):
                    self._apply_speech_event(event)
                # The sentinel also arrives during an orderly shutdown, which is
                # not the agent leaving. Only record a departure if the session
                # was still running -- otherwise every clean session would end
                # with a spurious "the agent left" in its record.
                if self.log.stop_reason is None:
                    self.log.record(AGENT_LEFT, ACTOR_AGENT)
                self._agent_gone.set()
                self._transition.set()
                return

            if self._agent_writer is not None:
                self._agent_writer.write(frame.pcm)
            for event in self.detector.push(frame.pcm, frame.t_ms):
                self._apply_speech_event(event)

    def _apply_speech_event(self, event) -> None:
        if event.kind == SPEECH_START:
            self._agent_speaking = True
            self._agent_speech_started_ms = event.t_ms
            self.log.record(
                AGENT_SPEECH_START,
                ACTOR_AGENT,
                t_ms=event.t_ms,
                declared_at_ms=event.declared_at_ms,
                detection_lag_ms=event.detection_lag_ms,
                energy=round(event.energy, 5),
            )
            self.log.begin_turn(ACTOR_AGENT, event.t_ms)
        elif event.kind == SPEECH_END:
            self._agent_speaking = False
            self.log.record(
                AGENT_SPEECH_END,
                ACTOR_AGENT,
                t_ms=event.t_ms,
                declared_at_ms=event.declared_at_ms,
                detection_lag_ms=event.detection_lag_ms,
            )
            self.log.end_turn(ACTOR_AGENT, event.t_ms)
        self._transition.set()

    def _drain_transcripts(self) -> bool:
        """Pull any agent transcript lines. Returns True if one was a repeat."""
        repeated = False
        while not self.transport.agent_text.empty():
            line = self.transport.agent_text.get_nowait()
            if self.selector.note_agent_line(line):
                repeated = True
            self.log.record(AGENT_TRANSCRIPT, ACTOR_AGENT, text=line)
        return repeated

    async def _wait_for_speaking(self, want: bool, timeout_ms: int) -> bool:
        """Block until the agent is/is not speaking, or the timeout expires."""
        deadline = self._elapsed_ms() + timeout_ms
        while self._agent_speaking != want:
            if self._agent_gone.is_set():
                return False
            remaining = deadline - self._elapsed_ms()
            if remaining <= 0:
                return False
            self._transition.clear()
            if not await race_timeout(
                self.scheduler, self._transition.wait(), remaining
            ):
                return False
        return True

    # -- the conversation ----------------------------------------------------

    def _wants_barge_in(self, turn: CandidateTurn) -> bool:
        """Decide whether this turn is spoken over the agent.

        Explicitly scripted interruptions always barge in. Beyond those, the
        persona's aggression is a per-turn probability drawn from a seeded
        generator, so a given seed reproduces a given conversation exactly --
        without which a benchmark that involves interruption could not be
        compared against itself between runs.
        """
        if turn.intent == "interrupting":
            return True
        return self.rng.random() < self.persona.interruption_aggression

    def _pause_before(self, turn: CandidateTurn) -> int:
        return (
            turn.delay_ms
            if turn.delay_ms is not None
            else self.persona.silence_before_reply_ms
        )

    async def _speak(self, turn: CandidateTurn, index: int, intended_barge_in: bool) -> None:
        clip = self.clip_source.clip_for(self.persona, turn, index)
        self._published.append(clip)

        started: Dict[str, int] = {}

        def on_air(t_ms: int) -> None:
            started["t"] = t_ms
            # The harness's live belief about whether the agent is still
            # talking. It lags the truth by up to the detector's silence
            # window, so it is recorded as evidence rather than treated as the
            # answer; the definitive overlap is resolved at session end.
            believed_speaking = self._agent_speaking
            self.log.record(
                CANDIDATE_SPEECH_START,
                ACTOR_CANDIDATE,
                t_ms=t_ms,
                intent=turn.intent,
                text=turn.text,
                intended_barge_in=intended_barge_in,
                agent_believed_speaking=believed_speaking,
                duration_ms=clip.duration_ms,
            )
            self.log.begin_turn(
                ACTOR_CANDIDATE,
                t_ms,
                intent=turn.intent,
                text=turn.text,
                intended_barge_in=intended_barge_in,
            )
            if intended_barge_in:
                into = self._agent_speech_started_ms
                self.log.record(
                    BARGE_IN,
                    ACTOR_CANDIDATE,
                    t_ms=t_ms,
                    agent_believed_speaking=believed_speaking,
                    into_agent_turn_ms=(None if into is None else t_ms - into),
                )

        await self.transport.publish(clip, on_air=on_air)
        end_ms = self._elapsed_ms()
        self.log.record(
            CANDIDATE_SPEECH_END,
            ACTOR_CANDIDATE,
            t_ms=end_ms,
            intent=turn.intent,
            spoken_ms=end_ms - started.get("t", end_ms),
        )
        self.log.end_turn(ACTOR_CANDIDATE, end_ms)

        if self.verbose:
            mark = "><" if intended_barge_in else "  "
            print(
                f"  [{end_ms / 1000:>6.1f}s] {mark} candidate "
                f"({turn.intent}) {turn.text[:56]}"
            )

    async def _converse(self) -> str:
        strikes = 0
        spoken = 0

        while True:
            if spoken >= self.config.max_turns:
                return STOP_TURN_CEILING
            if self._budget_left_ms() <= 0:
                return STOP_WALL_CLOCK
            if self._agent_gone.is_set():
                return STOP_AGENT_LEFT

            # Wait for the agent to say something. An interviewer speaks first,
            # so on the first pass this is waiting for the opening question.
            wait_ms = min(self.config.agent_silence_timeout_ms, self._budget_left_ms())
            heard = self._agent_speaking or await self._wait_for_speaking(True, wait_ms)
            if not heard:
                if self._agent_gone.is_set():
                    return STOP_AGENT_LEFT
                if self._budget_left_ms() <= 0:
                    return STOP_WALL_CLOCK
                strikes += 1
                self.log.record(
                    DEAD_AIR,
                    ACTOR_HARNESS,
                    waited_ms=wait_ms,
                    strike=strikes,
                    of=self.config.max_dead_air_strikes,
                )
                if self.verbose:
                    print(
                        f"  [{self._elapsed_ms() / 1000:>6.1f}s]    dead air "
                        f"({strikes}/{self.config.max_dead_air_strikes})"
                    )
                if strikes >= self.config.max_dead_air_strikes:
                    return STOP_AGENT_SILENT
                continue
            strikes = 0

            repeated = self._drain_transcripts()
            turn = self.selector.next_turn(agent_repeated=repeated)
            if turn is None:
                return STOP_SCRIPT_EXHAUSTED
            self.log.record(
                TURN_SELECTED,
                ACTOR_HARNESS,
                intent=turn.intent,
                text=turn.text,
                after_agent_repeat=repeated,
            )

            barge_in = self._wants_barge_in(turn)

            # At most one interruption per agent utterance. Without this, an
            # aggressive persona fires again the instant the previous turn ends:
            # the barge-in window for the utterance has already passed, so the
            # wait computes as zero and the candidate machine-guns several turns
            # into one utterance with no agent speech between them. That is not
            # a behaviour any real speaker has, and it buries the interruption
            # being tested under noise.
            current_start = self._agent_speech_started_ms
            if (
                barge_in
                and current_start is not None
                and current_start == self._last_barged_start_ms
            ):
                barge_in = False

            if barge_in:
                self._last_barged_start_ms = current_start
                # Cut in partway through the agent's utterance rather than at
                # its first syllable: interrupting instantly is a different and
                # much rarer failure than interrupting mid-sentence.
                #
                # Measured from when the agent *started speaking*, not from now.
                # Sleeping a fixed window from the decision point overshoots
                # whenever the decision is made late in an utterance -- and the
                # decision is routinely late, because the detector only reports
                # the previous utterance ending after its silence window has
                # elapsed. That overshoot was silently turning interruptions
                # into ordinary replies while still labelling them barge-ins.
                started = self._agent_speech_started_ms
                target = (
                    self._elapsed_ms() if started is None else started
                ) + self.config.barge_in_after_ms
                await self.scheduler.sleep_ms(
                    min(
                        max(target - self._elapsed_ms(), 0),
                        max(self._budget_left_ms(), 0),
                    )
                )
            else:
                await self._wait_for_speaking(
                    False, min(self.config.agent_silence_timeout_ms,
                               max(self._budget_left_ms(), 0))
                )
                self._drain_transcripts()
                await self.scheduler.sleep_ms(
                    min(self._pause_before(turn), max(self._budget_left_ms(), 0))
                )

            if self._budget_left_ms() <= 0:
                return STOP_WALL_CLOCK

            await self._speak(turn, len(self.selector.used), barge_in)
            spoken += 1

    # -- lifecycle -----------------------------------------------------------

    async def run(self) -> SessionLog:
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        agent_wav = self.audio_dir / f"{self.log.session_id}_agent.wav"
        self._agent_writer = StreamingWavWriter(agent_wav, self.config.sample_rate)

        consumer: Optional[asyncio.Task] = None
        stop_reason = STOP_ERROR
        error: Optional[str] = None

        try:
            await self.transport.connect()
            consumer = asyncio.create_task(
                self._consume_agent_audio(), name="agent-audio"
            )

            joined = await self.transport.wait_for_agent(
                self.config.agent_join_timeout_ms
            )
            if not joined:
                stop_reason = STOP_AGENT_NEVER_JOINED
            else:
                self.log.record(AGENT_JOINED, ACTOR_AGENT)
                if self.verbose:
                    print(f"  [{self._elapsed_ms() / 1000:>6.1f}s]    agent joined")
                # The whole conversation is wrapped in the wall-clock budget as
                # well as checking it internally. The internal checks keep the
                # stop reason honest; this one guarantees termination even if a
                # future change introduces a path that forgets to check.
                converse = asyncio.ensure_future(self._converse())
                if await race_timeout(
                    self.scheduler, converse, self.config.max_wall_clock_ms
                ):
                    stop_reason = converse.result()
                else:
                    stop_reason = STOP_WALL_CLOCK
        except TransportError as exc:
            stop_reason, error = STOP_ERROR, str(exc)
            self.log.warn("transport error", detail=str(exc))
        except asyncio.CancelledError:
            stop_reason, error = STOP_ERROR, "cancelled"
            raise
        finally:
            self.log.finish(stop_reason, error)
            await self.transport.close()
            if consumer is not None:
                # close() queues the end-of-audio sentinel. Letting the consumer
                # reach it lets the detector flush an utterance still open, so a
                # session cut short by a timeout still has a closed final agent
                # segment rather than a dangling one. Real seconds deliberately:
                # this is draining a queue, not waiting out session time.
                try:
                    await asyncio.wait_for(consumer, timeout=10)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    consumer.cancel()
                    try:
                        await consumer
                    except (asyncio.CancelledError, Exception):
                        pass
            if self._agent_writer is not None:
                self._agent_writer.close()

            # Every agent segment is now closed, so true speech overlap can be
            # settled from timestamps that were all recorded live.
            self.log.resolve_overlaps()

            self.log.config["clip_source"] = self.clip_source.to_dict()
            self.log.config["detector"] = self.detector.to_dict()
            self.log.config["agent_audio_wav"] = str(agent_wav)
            if self._published:
                candidate_wav = (
                    self.audio_dir / f"{self.log.session_id}_candidate.wav"
                )
                concat(self._published).to_wav(candidate_wav)
                self.log.config["candidate_audio_wav"] = str(candidate_wav)

        return self.log


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def build_transport(
    kind: str,
    config: SessionConfig,
    scheduler: Scheduler,
    loopback_agent: Optional[LoopbackAgent] = None,
) -> Transport:
    """Pick a transport. ``loopback`` needs no server, key or account."""
    if kind == "loopback":
        return LoopbackTransport(
            scheduler=scheduler,
            agent=loopback_agent or LoopbackAgent(),
            sample_rate=config.sample_rate,
            frame_ms=config.frame_ms,
            seed=config.seed,
        )
    if kind == "livekit":
        return LiveKitTransport(
            config=LiveKitConfig(
                url=os.environ.get("LIVEKIT_URL", ""),
                api_key=os.environ.get("LIVEKIT_API_KEY", ""),
                api_secret=os.environ.get("LIVEKIT_API_SECRET", ""),
                room=config.room,
                agent_identity_contains=os.environ.get("HARNESS_AGENT_IDENTITY", ""),
            ),
            scheduler=scheduler,
            sample_rate=config.sample_rate,
            frame_ms=config.frame_ms,
        )
    raise ValueError(f"unknown transport {kind!r}; use 'loopback' or 'livekit'")


def build_clip_source(
    sample_rate: int, cache_dir: Path, seed: int, use_tts: bool
) -> ClipSource:
    """Real synthesis when a key is configured and asked for, silence-shaped audio otherwise."""
    tts = None
    if use_tts:
        from persona_tts import CartesiaTTSBackend

        tts = CartesiaTTSBackend(
            api_key=os.environ.get("CARTESIA_API_KEY", ""),
            voice_id=os.environ.get("CARTESIA_VOICE_ID", ""),
        )
    return ClipSource(cache_dir=cache_dir, sample_rate=sample_rate, tts=tts, seed=seed)


async def run_session(
    config: SessionConfig,
    transport_kind: str = "loopback",
    clock: str = "real",
    use_tts: bool = False,
    transcript_dir: Path = DEFAULT_TRANSCRIPT_DIR,
    audio_dir: Path = DEFAULT_AUDIO_DIR,
    loopback_agent: Optional[LoopbackAgent] = None,
    verbose: bool = True,
) -> SessionLog:
    """Run one session end to end and write its artifacts.

    ``clock="virtual"`` simulates time instead of spending it, turning a
    three-minute conversation into a fraction of a second. It is available only
    against the loopback transport -- a real room runs at the speed it runs at --
    and the session it produces is marked so that nothing downstream mistakes it
    for a measurement.
    """
    if clock not in ("real", "virtual"):
        raise ValueError(f"unknown clock {clock!r}; use 'real' or 'virtual'")
    if clock == "virtual" and transport_kind != "loopback":
        raise ValueError(
            "virtual time only applies to the loopback transport. A real room "
            "runs in real time, and simulating the harness clock against it "
            "would make every measurement wrong."
        )

    virtual = VirtualScheduler() if clock == "virtual" else None
    scheduler: Scheduler = virtual or RealScheduler()
    transport = build_transport(transport_kind, config, scheduler, loopback_agent)
    clip_source = build_clip_source(
        config.sample_rate, audio_dir / "cache", config.seed, use_tts
    )
    session = AudioSession(
        config=config,
        transport=transport,
        scheduler=scheduler,
        clip_source=clip_source,
        transcript_dir=transcript_dir,
        audio_dir=audio_dir,
        verbose=verbose,
    )
    if virtual is not None:
        virtual.start()
    try:
        log = await session.run()
    finally:
        if virtual is not None:
            await virtual.stop()
    path = log.write(transcript_dir)
    if verbose:
        print(f"\n{log.summary_line()}\n  record: {path}")
    return log


def run_audio_session(
    persona_name: str, room_name: str, out_dir: Path, **kwargs
) -> Dict:
    """Synchronous convenience wrapper. Returns the session record."""
    config = SessionConfig(persona=persona_name, room=room_name)
    log = asyncio.run(
        run_session(config, transcript_dir=out_dir, audio_dir=out_dir, **kwargs)
    )
    return log.to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio_loop",
        description="Drive a persona through a live agent and capture both sides.",
    )
    parser.add_argument("--persona", choices=sorted(PERSONAS), default="cooperative")
    parser.add_argument("--room", default="harness")
    parser.add_argument(
        "--transport",
        choices=("loopback", "livekit"),
        default="loopback",
        help="loopback simulates an agent in-process and needs nothing",
    )
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--max-seconds", type=int, default=180)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--clock",
        choices=("real", "virtual"),
        default="real",
        help="virtual simulates time instead of spending it (loopback only)",
    )
    parser.add_argument(
        "--tts",
        action="store_true",
        help="render real speech via the TTS provider instead of shaped tones",
    )
    parser.add_argument("--transcript-dir", type=Path, default=DEFAULT_TRANSCRIPT_DIR)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    config = SessionConfig(
        persona=args.persona,
        room=args.room,
        max_turns=args.max_turns,
        max_wall_clock_ms=args.max_seconds * 1000,
        seed=args.seed,
    )

    try:
        log = asyncio.run(
            run_session(
                config,
                transport_kind=args.transport,
                clock=args.clock,
                use_tts=args.tts,
                transcript_dir=args.transcript_dir,
                audio_dir=args.audio_dir,
                verbose=not args.quiet,
            )
        )
    except (TransportError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    return 0 if log.stop_reason in (STOP_SCRIPT_EXHAUSTED, STOP_TURN_CEILING) else 1


if __name__ == "__main__":
    raise SystemExit(main())
