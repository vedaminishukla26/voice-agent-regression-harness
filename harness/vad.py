"""Energy-based speech detection over arriving audio frames.

The loop has to know when the agent starts and stops talking. That question
sounds simple and is the single largest source of error in everything week 3
measures, so the reasoning is written down here rather than buried in constants.

**Why energy and not a model.** A neural VAD would classify speech-versus-noise
better. It would also add a model dependency, a download, and a load time to a
layer whose actual job is timestamping, not classification. The audio arriving
here is a single remote participant on a clean WebRTC track -- not a microphone
in a room -- so the discrimination problem is easy and the timing problem is
not. If the agent under test is ever driven from a noisy source, this is the
module to replace, and it sits behind a small enough surface to replace.

**Why the timestamps are retroactive.** A detector cannot declare speech until
it has seen evidence, and it cannot declare silence until enough silence has
passed to distinguish a pause from the end of a turn. Reporting the moment of
*declaration* would therefore push every start later and every end later by the
size of those windows -- a fixed bias of a few hundred milliseconds baked
silently into every latency number downstream.

So a declaration reports the timestamp at which the transition actually
happened, reconstructed from the frames that produced it. The detector is
allowed to be slow to decide; it is not allowed to be wrong about when.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from audio_io import DEFAULT_FRAME_MS, rms

SPEECH_START = "speech_start"
SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VadConfig:
    """Tuning for :class:`SpeechDetector`."""

    # Frames at or above this RMS count as speech. 0.02 is roughly -34 dBFS:
    # comfortably above codec noise on a silent track, well below any real
    # utterance. Synthesised agent speech is typically far louder still.
    threshold_rms: float = 0.02

    # Consecutive speech evidence needed to declare a start. Long enough to
    # ignore a click or a single-frame artefact, short enough not to clip the
    # onset of a word.
    start_ms: int = 60

    # Consecutive silence needed to declare an end. This is the important one:
    # it must exceed the longest gap *within* an utterance, or one sentence gets
    # reported as several turns. Natural inter-word gaps run to about 200 ms and
    # inter-clause pauses further, so 400 ms is deliberately generous.
    #
    # Note this is the harness's own endpointing threshold, and it is not the
    # agent's. The harness is measuring when the agent *actually* stopped
    # emitting audio, not deciding when to reply.
    end_ms: int = 400

    frame_ms: int = DEFAULT_FRAME_MS

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_rms < 1.0:
            raise ValueError(
                f"threshold_rms must be in (0.0, 1.0), got {self.threshold_rms}"
            )
        if self.frame_ms <= 0:
            raise ValueError(f"frame_ms must be positive, got {self.frame_ms}")
        for field_name in ("start_ms", "end_ms"):
            value = getattr(self, field_name)
            if value < self.frame_ms:
                raise ValueError(
                    f"{field_name} ({value} ms) is shorter than one frame "
                    f"({self.frame_ms} ms), so it could never be satisfied"
                )

    @property
    def start_frames(self) -> int:
        return max(1, round(self.start_ms / self.frame_ms))

    @property
    def end_frames(self) -> int:
        return max(1, round(self.end_ms / self.frame_ms))


@dataclass(frozen=True)
class SpeechEvent:
    """A speech boundary, timestamped at when it happened, not when it was seen."""

    kind: str
    t_ms: int              # reconstructed moment of the transition
    declared_at_ms: int    # when the detector could first be sure
    energy: float

    @property
    def detection_lag_ms(self) -> int:
        """How much later than the truth the detector reached its conclusion.

        Reported rather than hidden: it is the resolution limit of every
        derived latency, and a reader of the metrics deserves to see it.
        """
        return self.declared_at_ms - self.t_ms


class SpeechDetector:
    """Streaming speech/silence segmenter for one audio source.

    Frames are pushed in arrival order with the wall-clock offset at which they
    arrived. Each push returns the boundary events, if any, that this frame
    settled -- normally none.
    """

    def __init__(self, config: Optional[VadConfig] = None) -> None:
        self.config = config or VadConfig()
        self.is_speaking = False
        self.frames_seen = 0
        self.speech_ms = 0
        self.silence_ms = 0

        # Run lengths of the current above/below-threshold streak.
        self._speech_run = 0
        self._silence_run = 0

        # Start time of the current streak, so a declaration can point back at
        # the frame where the transition truly began.
        self._speech_run_started_ms = 0
        self._silence_run_started_ms = 0

        self._last_energy = 0.0

    def push(self, pcm: bytes, t_ms: int) -> List[SpeechEvent]:
        """Feed one frame that arrived at ``t_ms``. Returns any settled events."""
        energy = rms(pcm)
        self._last_energy = energy
        self.frames_seen += 1

        loud = energy >= self.config.threshold_rms
        if loud:
            self.speech_ms += self.config.frame_ms
            if self._speech_run == 0:
                self._speech_run_started_ms = t_ms
            self._speech_run += 1
            self._silence_run = 0
        else:
            self.silence_ms += self.config.frame_ms
            if self._silence_run == 0:
                self._silence_run_started_ms = t_ms
            self._silence_run += 1
            self._speech_run = 0

        events: List[SpeechEvent] = []
        if not self.is_speaking and self._speech_run >= self.config.start_frames:
            self.is_speaking = True
            events.append(
                SpeechEvent(
                    kind=SPEECH_START,
                    t_ms=self._speech_run_started_ms,
                    declared_at_ms=t_ms,
                    energy=energy,
                )
            )
        elif self.is_speaking and self._silence_run >= self.config.end_frames:
            self.is_speaking = False
            events.append(
                SpeechEvent(
                    kind=SPEECH_END,
                    t_ms=self._silence_run_started_ms,
                    declared_at_ms=t_ms,
                    energy=energy,
                )
            )
        return events

    def flush(self, t_ms: int) -> List[SpeechEvent]:
        """Close an utterance still open when the stream ends.

        A session that ends mid-agent-turn -- hit the wall-clock timeout, or the
        agent simply never stopped -- would otherwise leave an unterminated
        speech segment that the metrics layer has to guess about. Better to
        close it explicitly and let the timestamp say where.
        """
        if not self.is_speaking:
            return []
        self.is_speaking = False
        return [
            SpeechEvent(
                kind=SPEECH_END,
                t_ms=t_ms,
                declared_at_ms=t_ms,
                energy=self._last_energy,
            )
        ]

    @property
    def observed_ms(self) -> int:
        return self.frames_seen * self.config.frame_ms

    def to_dict(self) -> dict:
        """Summary for the session record."""
        return {
            "frames_seen": self.frames_seen,
            "observed_ms": self.observed_ms,
            "speech_ms": self.speech_ms,
            "silence_ms": self.silence_ms,
            "threshold_rms": self.config.threshold_rms,
            "start_ms": self.config.start_ms,
            "end_ms": self.config.end_ms,
        }
