"""Getting audio to an agent and back.

The loop above this module does not know what a room is. It knows how to hand
over a clip, how to be told that agent audio arrived, and how to be told the
agent has gone. That is the whole surface, and it exists so that two very
different things can sit underneath it:

:class:`LiveKitTransport`
    A real room. The agent under test is a *separate process* that joins the
    same room -- this harness never imports, configures or runs it. That is a
    deliberate boundary: it means the harness works against any agent on the
    platform, and it means no agent's prompt, model configuration or source ever
    needs to be present in this repository to test it.

:class:`LoopbackTransport`
    A simulated agent with configurable join delay, response latency, utterance
    length and barge-in behaviour. No server, no account, no network, no keys.
    This is what makes the week 2 control flow testable and what lets anyone
    clone the repository and watch the loop run without signing up for anything.

The LiveKit SDK is imported lazily, inside :meth:`LiveKitTransport.connect`, so
the offline path never pays for it and never requires it to be installed.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Protocol

from audio_io import (
    DEFAULT_FRAME_MS,
    DEFAULT_SAMPLE_RATE,
    PcmClip,
    estimate_duration_ms,
    synthesize_speech_like,
)
from session_log import Scheduler, race_timeout

# Sentinel pushed onto the audio queue when the agent's track ends, so a
# consumer blocked on the queue wakes up instead of waiting out its timeout.
END_OF_AUDIO = None


class TransportError(RuntimeError):
    """Raised when the transport cannot be established or has failed."""


@dataclass(frozen=True)
class AgentFrame:
    """One frame of audio from the agent, stamped when it arrived."""

    pcm: bytes
    sample_rate: int
    t_ms: int


class Transport(Protocol):
    """What the audio loop needs from the world."""

    name: str
    sample_rate: int
    agent_audio: "asyncio.Queue[Optional[AgentFrame]]"
    agent_text: "asyncio.Queue[str]"

    async def connect(self) -> None: ...

    async def wait_for_agent(self, timeout_ms: int) -> bool: ...

    async def publish(
        self, clip: PcmClip, on_air: Optional[Callable[[int], None]] = None
    ) -> int: ...

    async def close(self) -> None: ...

    @property
    def agent_present(self) -> bool: ...


# ---------------------------------------------------------------------------
# Offline loopback
# ---------------------------------------------------------------------------


@dataclass
class LoopbackAgent:
    """Behaviour of the simulated agent.

    Every field is a thing a real agent gets wrong in a way worth reproducing
    deterministically, which is the point: the offline transport is not a mock
    that always succeeds, it is a configurable opponent.
    """

    # How long before the agent joins. Non-zero so the join wait is exercised.
    join_delay_ms: int = 400

    # Whether the agent speaks first, as an interviewer normally would.
    greets: bool = True

    # End of candidate speech to start of agent speech. This is the number a
    # real harness run exists to measure, so the simulator has to have one.
    response_latency_ms: int = 900

    # Length of each agent utterance, cycled in order.
    utterance_ms: List[int] = field(default_factory=lambda: [2_600, 3_400, 2_100])

    # Whether the agent stops talking when the candidate talks over it. False
    # models an agent that ploughs on through an interruption -- a real and
    # common failure that the edge_case persona is built to provoke.
    interruptible: bool = True

    # How long after being interrupted the agent actually falls silent.
    barge_in_stop_ms: int = 250

    # Optional canned transcript lines, delivered as the agent starts each turn.
    transcripts: List[str] = field(default_factory=list)

    # Agent stops responding after this many turns. None means never.
    goes_silent_after_turn: Optional[int] = None

    def __post_init__(self) -> None:
        if self.join_delay_ms < 0:
            raise ValueError("join_delay_ms must not be negative")
        if self.response_latency_ms < 0:
            raise ValueError("response_latency_ms must not be negative")
        if not self.utterance_ms or any(ms <= 0 for ms in self.utterance_ms):
            raise ValueError("utterance_ms must be a non-empty list of positive values")


class LoopbackTransport:
    """A simulated agent in-process. No server, no network, no keys.

    Emits a continuous audio stream exactly as a real subscribed track does --
    silence frames when the agent is not talking, not an absence of frames.
    Getting that right matters: a detector that only ever sees audio when
    someone is speaking is not being tested on the case it exists to handle.
    """

    def __init__(
        self,
        scheduler: Scheduler,
        agent: Optional[LoopbackAgent] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
        seed: int = 0,
    ) -> None:
        self.name = "loopback"
        self.scheduler = scheduler
        self.agent = agent or LoopbackAgent()
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.seed = seed

        self.agent_audio: "asyncio.Queue[Optional[AgentFrame]]" = asyncio.Queue()
        self.agent_text: "asyncio.Queue[str]" = asyncio.Queue()

        self._joined = asyncio.Event()
        self._closed = False
        self._pump_task: Optional[asyncio.Task] = None

        # Simulated agent state, all in scheduler milliseconds.
        self._speaking_until_ms: Optional[int] = None
        self._speak_at_ms: Optional[int] = None
        self._candidate_speaking = False
        self._agent_turns = 0
        self._utterance_index = 0

        speech = synthesize_speech_like(2_000, seed=seed + 7, sample_rate=sample_rate)
        self._speech_frames = list(speech.frames(frame_ms))
        self._silence_frame = b"\x00" * (
            int(sample_rate * frame_ms / 1000) * 2
        )
        self._speech_cursor = 0

    @property
    def agent_present(self) -> bool:
        return self._joined.is_set() and not self._closed

    async def connect(self) -> None:
        self._pump_task = asyncio.create_task(self._pump(), name="loopback-pump")

    async def wait_for_agent(self, timeout_ms: int) -> bool:
        return await race_timeout(self.scheduler, self._joined.wait(), timeout_ms)

    async def publish(
        self, clip: PcmClip, on_air: Optional[Callable[[int], None]] = None
    ) -> int:
        """Play a candidate clip into the simulation in scheduler time."""
        if self._closed:
            raise TransportError("publish on a closed transport")
        if on_air is not None:
            on_air(self.scheduler.now_ms())

        self._candidate_speaking = True
        # A candidate talking over the agent is exactly what barge-in is. The
        # simulated agent reacts to it here rather than the harness asserting
        # what it would have done.
        if self._speaking_until_ms is not None and self.agent.interruptible:
            self._speaking_until_ms = min(
                self._speaking_until_ms,
                self.scheduler.now_ms() + self.agent.barge_in_stop_ms,
            )

        await self.scheduler.sleep_ms(clip.duration_ms)
        self._candidate_speaking = False

        if self.agent.goes_silent_after_turn is None or (
            self._agent_turns < self.agent.goes_silent_after_turn
        ):
            self._speak_at_ms = self.scheduler.now_ms() + self.agent.response_latency_ms
        return clip.duration_ms

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
        await self.agent_audio.put(END_OF_AUDIO)

    def _next_frame(self, speaking: bool) -> bytes:
        if not speaking:
            return self._silence_frame
        frame = self._speech_frames[self._speech_cursor % len(self._speech_frames)]
        self._speech_cursor += 1
        return frame

    def _begin_agent_turn(self, now_ms: int) -> None:
        duration = self.agent.utterance_ms[
            self._utterance_index % len(self.agent.utterance_ms)
        ]
        self._utterance_index += 1
        self._agent_turns += 1
        self._speaking_until_ms = now_ms + duration
        self._speak_at_ms = None
        if self.agent.transcripts:
            line = self.agent.transcripts[
                (self._agent_turns - 1) % len(self.agent.transcripts)
            ]
            self.agent_text.put_nowait(line)

    async def _pump(self) -> None:
        """Emit one frame every ``frame_ms``, speaking or silent."""
        await self.scheduler.sleep_ms(self.agent.join_delay_ms)
        self._joined.set()
        if self.agent.greets:
            self._speak_at_ms = self.scheduler.now_ms()

        while not self._closed:
            now = self.scheduler.now_ms()

            if self._speaking_until_ms is not None and now >= self._speaking_until_ms:
                self._speaking_until_ms = None
            if (
                self._speaking_until_ms is None
                and self._speak_at_ms is not None
                and now >= self._speak_at_ms
            ):
                self._begin_agent_turn(now)

            speaking = self._speaking_until_ms is not None
            await self.agent_audio.put(
                AgentFrame(
                    pcm=self._next_frame(speaking),
                    sample_rate=self.sample_rate,
                    t_ms=now,
                )
            )
            await self.scheduler.sleep_ms(self.frame_ms)


# ---------------------------------------------------------------------------
# LiveKit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveKitConfig:
    """Connection details. Values come from the environment, never from source."""

    url: str
    api_key: str
    api_secret: str
    room: str
    identity: str = "harness-candidate"
    participant_name: str = "Simulated candidate"

    # Substring that identifies the agent among remote participants. Empty means
    # "the first remote participant that publishes audio", which is the right
    # default in a room containing only the harness and one agent.
    agent_identity_contains: str = ""

    # Name of an agent to dispatch into the room on join. Leave empty for a
    # worker that registers without one.
    #
    # This is not an optional nicety. A worker started with `agent_name` set is
    # excluded from automatic dispatch -- the server will not hand it a room
    # just because one appeared. Joining such a room and waiting produces a
    # perfectly healthy-looking session that reports the agent never joined,
    # with nothing anywhere saying why. Requesting the dispatch explicitly is
    # the only thing that brings the agent in.
    agent_name: str = ""

    # JSON handed to the agent as job metadata. Agents commonly take their
    # entire configuration this way -- prompt, language, voice, feature flags --
    # so this is usually how a session is shaped, and it is read from a file
    # outside the repository precisely because its contents tend to be private.
    agent_metadata: str = ""

    def __post_init__(self) -> None:
        missing = [
            name
            for name in ("url", "api_key", "api_secret", "room")
            if not getattr(self, name)
        ]
        if missing:
            raise TransportError(
                "missing LiveKit settings: "
                + ", ".join(missing)
                + ". Set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET and pass "
                "--room, or use --transport loopback to run with no server."
            )


def build_access_token(api, config: LiveKitConfig) -> str:
    """Mint the harness's join token, including any agent dispatch request.

    Separate from :meth:`LiveKitTransport.connect` so the dispatch request can
    be verified without a server. Whether the agent is asked for is the single
    difference between a working run and a session that reports the agent never
    joined, and that is not a thing to discover only against live infrastructure.

    ``api`` is passed in rather than imported so the module continues to load
    without the LiveKit SDK installed.
    """
    builder = (
        api.AccessToken(config.api_key, config.api_secret)
        .with_identity(config.identity)
        .with_name(config.participant_name)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=config.room,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )
    if config.agent_name:
        # Ask for the dispatch as part of joining rather than as a separate API
        # call afterwards. The room is created by this join, so the request
        # cannot arrive before the room exists, nor be left orphaned if the join
        # fails -- both of which are possible when the two are separate steps.
        builder = builder.with_room_config(
            api.RoomConfiguration(
                agents=[
                    api.RoomAgentDispatch(
                        agent_name=config.agent_name,
                        metadata=config.agent_metadata,
                    )
                ]
            )
        )
    return builder.to_jwt()


class LiveKitTransport:
    """Joins a room as the candidate and captures whatever the agent says.

    The agent under test is not started, configured or imported here. It is a
    separate process that joins the same room by whatever means it normally
    would. The harness only has to be in the room first and listen.
    """

    def __init__(
        self,
        config: LiveKitConfig,
        scheduler: Scheduler,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_ms: int = DEFAULT_FRAME_MS,
    ) -> None:
        self.name = "livekit"
        self.config = config
        self.scheduler = scheduler
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms

        self.agent_audio: "asyncio.Queue[Optional[AgentFrame]]" = asyncio.Queue()
        self.agent_text: "asyncio.Queue[str]" = asyncio.Queue()

        self._room: Any = None
        self._source: Any = None
        self._rtc: Any = None
        self._joined = asyncio.Event()
        self._agent_identity: Optional[str] = None
        self._stream_tasks: List[asyncio.Task] = []
        self._closed = False

    @property
    def agent_present(self) -> bool:
        return self._agent_identity is not None and not self._closed

    def _is_agent(self, identity: str) -> bool:
        if identity == self.config.identity:
            return False
        return self.config.agent_identity_contains in identity

    async def connect(self) -> None:
        try:
            from livekit import api, rtc
        except ImportError as exc:  # pragma: no cover - depends on install state
            raise TransportError(
                "the livekit SDK is not installed. Install it with "
                "`pip install -r requirements.txt`, or run with "
                "--transport loopback, which needs nothing."
            ) from exc

        self._rtc = rtc
        token = build_access_token(api, self.config)

        self._room = rtc.Room()
        # Handlers are registered before connecting. An agent already in the
        # room when we arrive would otherwise publish its track during the
        # connect handshake and never be seen.
        self._room.on("track_subscribed", self._on_track_subscribed)
        self._room.on("participant_disconnected", self._on_participant_disconnected)
        self._room.on("transcription_received", self._on_transcription)
        self._room.on("disconnected", self._on_disconnected)

        try:
            await self._room.connect(self.config.url, token)
        except Exception as exc:
            raise TransportError(
                f"could not connect to {self.config.url}: {exc}"
            ) from exc

        # A small publish queue keeps submitted audio close to aired audio. The
        # SDK default buffers a full second, which would put every candidate
        # timestamp a second ahead of when the sound actually left.
        self._source = rtc.AudioSource(self.sample_rate, 1, queue_size_ms=120)
        track = rtc.LocalAudioTrack.create_audio_track("candidate", self._source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self._room.local_participant.publish_track(track, options)

        for participant in self._room.remote_participants.values():
            if self._is_agent(participant.identity):
                self._note_agent(participant.identity)

    def _note_agent(self, identity: str) -> None:
        if self._agent_identity is None:
            self._agent_identity = identity
            self._joined.set()

    def _on_track_subscribed(self, track: Any, publication: Any, participant: Any) -> None:
        if track.kind != self._rtc.TrackKind.KIND_AUDIO:
            return
        if not self._is_agent(participant.identity):
            return
        self._note_agent(participant.identity)
        self._stream_tasks.append(
            asyncio.create_task(self._consume(track), name="livekit-audio")
        )

    def _on_participant_disconnected(self, participant: Any) -> None:
        if participant.identity == self._agent_identity:
            self._agent_identity = None
            self.agent_audio.put_nowait(END_OF_AUDIO)

    def _on_disconnected(self, *_: Any) -> None:
        self.agent_audio.put_nowait(END_OF_AUDIO)

    def _on_transcription(self, segments: Any, *_: Any) -> None:
        """Capture agent transcripts if the agent happens to publish them.

        Strictly best effort. Many agents publish transcriptions and many do
        not, so nothing in the loop may depend on this arriving -- it enriches
        the record and informs turn selection when present, and its absence
        changes no measurement.
        """
        try:
            for segment in segments:
                text = getattr(segment, "text", "")
                final = getattr(segment, "final", True)
                if text and final:
                    self.agent_text.put_nowait(text)
        except TypeError:
            pass

    async def _consume(self, track: Any) -> None:
        """Forward one agent audio track onto the queue, stamped on arrival."""
        stream = self._rtc.AudioStream(
            track,
            sample_rate=self.sample_rate,
            num_channels=1,
            frame_size_ms=self.frame_ms,
        )
        try:
            async for event in stream:
                frame = event.frame
                await self.agent_audio.put(
                    AgentFrame(
                        pcm=bytes(frame.data),
                        sample_rate=frame.sample_rate,
                        t_ms=self.scheduler.now_ms(),
                    )
                )
        except asyncio.CancelledError:
            raise
        finally:
            await stream.aclose()

    async def wait_for_agent(self, timeout_ms: int) -> bool:
        return await race_timeout(self.scheduler, self._joined.wait(), timeout_ms)

    async def publish(
        self, clip: PcmClip, on_air: Optional[Callable[[int], None]] = None
    ) -> int:
        if self._source is None:
            raise TransportError("publish before connect")
        if clip.sample_rate != self.sample_rate:
            raise TransportError(
                f"clip is {clip.sample_rate} Hz but the transport publishes at "
                f"{self.sample_rate} Hz. Render audio at the transport rate "
                "rather than resampling it here."
            )

        samples_per_frame = clip.samples_per_frame(self.frame_ms)
        first = True
        for chunk in clip.frames(self.frame_ms):
            frame = self._rtc.AudioFrame(
                data=chunk,
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=samples_per_frame,
            )
            await self._source.capture_frame(frame)
            if first and on_air is not None:
                on_air(self.scheduler.now_ms())
                first = False

        # capture_frame returns once the frame is queued, not once it is heard.
        # Waiting for playout is what makes the end-of-utterance timestamp mean
        # "the candidate stopped talking" rather than "the harness stopped
        # typing", and end-of-utterance is the anchor for every latency below.
        await self._source.wait_for_playout()
        return clip.duration_ms

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for task in self._stream_tasks:
            task.cancel()
        for task in self._stream_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        if self._room is not None:
            try:
                await self._room.disconnect()
            except Exception:
                pass
        await self.agent_audio.put(END_OF_AUDIO)


def estimate_clip_for(text: str, speech_rate: float, seed: int) -> PcmClip:
    """Offline stand-in clip of realistic length for a piece of text.

    Used when no TTS key is configured. The loop still needs audio of plausible
    duration or the turn-taking logic is never put under any pressure.
    """
    return synthesize_speech_like(
        estimate_duration_ms(text, speech_rate), seed=seed
    )
