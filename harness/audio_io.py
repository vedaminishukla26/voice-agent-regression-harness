"""PCM primitives for the audio loop.

Week 2 needs to push audio into a live room and reason about audio coming back
out of it. Both directions want the same thing: signed 16-bit little-endian PCM,
mono, at a known sample rate, sliced into fixed-duration frames.

Everything here is standard library. The two obvious dependencies -- an MP3
decoder and a resampler -- are avoided rather than vendored:

*No decoder.* The TTS layer is asked for raw PCM directly instead of MP3, so
nothing ever has to decode a compressed container. MP3 remains available for
human ear-checks; it is simply not what the loop consumes.

*No resampler.* Audio is requested at the transport's native rate, so there is
nothing to resample. ``audioop`` would have been the stdlib answer and it was
removed in Python 3.13, which is a fair illustration of why not depending on it
was worth the trouble.
"""

from __future__ import annotations

import array
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List

# LiveKit's native rate. Choosing it everywhere is what lets the no-resampler
# rule hold: the TTS provider is asked for this rate, the transport publishes at
# this rate, and captured audio arrives at this rate.
DEFAULT_SAMPLE_RATE = 48_000

# 20 ms is the conventional WebRTC frame. It also sets the resolution of every
# timing measurement built on frame arrival, so it is a floor on how precisely
# week 3 can attribute latency.
DEFAULT_FRAME_MS = 20

BYTES_PER_SAMPLE = 2
INT16_MAX = 32_767


class AudioError(ValueError):
    """Raised when audio data is not in the one format this module accepts."""


@dataclass(frozen=True)
class PcmClip:
    """Mono signed-16-bit little-endian PCM at a known sample rate."""

    data: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise AudioError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.channels != 1:
            raise AudioError(
                f"only mono is supported, got {self.channels} channels. The loop "
                "measures one speaker at a time; stereo would only hide that."
            )
        if len(self.data) % BYTES_PER_SAMPLE:
            raise AudioError(
                f"PCM length {len(self.data)} is not a whole number of 16-bit samples"
            )

    @property
    def sample_count(self) -> int:
        return len(self.data) // BYTES_PER_SAMPLE

    @property
    def duration_ms(self) -> int:
        return int(round(self.sample_count * 1000 / self.sample_rate))

    def samples_per_frame(self, frame_ms: int = DEFAULT_FRAME_MS) -> int:
        if frame_ms <= 0:
            raise AudioError(f"frame_ms must be positive, got {frame_ms}")
        return int(self.sample_rate * frame_ms / 1000)

    def frames(self, frame_ms: int = DEFAULT_FRAME_MS) -> Iterator[bytes]:
        """Yield fixed-size frames, zero-padding the last one.

        Padding rather than truncating: a short final frame would be rejected by
        the transport, and dropping it would silently shorten every utterance by
        up to one frame. Trailing silence is the harmless option.
        """
        step = self.samples_per_frame(frame_ms) * BYTES_PER_SAMPLE
        for start in range(0, len(self.data), step):
            chunk = self.data[start : start + step]
            if len(chunk) < step:
                chunk = chunk + b"\x00" * (step - len(chunk))
            yield chunk

    def frame_count(self, frame_ms: int = DEFAULT_FRAME_MS) -> int:
        if not self.data:
            return 0
        step = self.samples_per_frame(frame_ms) * BYTES_PER_SAMPLE
        return math.ceil(len(self.data) / step)

    def to_wav(self, path: Path) -> Path:
        """Write a real WAV file, for listening to what was actually sent."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(self.channels)
            handle.setsampwidth(BYTES_PER_SAMPLE)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.data)
        return path

    @classmethod
    def from_wav(cls, path: Path) -> "PcmClip":
        with wave.open(str(path), "rb") as handle:
            if handle.getsampwidth() != BYTES_PER_SAMPLE:
                raise AudioError(
                    f"{path}: expected 16-bit PCM, got {handle.getsampwidth() * 8}-bit"
                )
            if handle.getnchannels() != 1:
                raise AudioError(f"{path}: expected mono audio")
            return cls(
                data=handle.readframes(handle.getnframes()),
                sample_rate=handle.getframerate(),
            )

    @classmethod
    def silence(
        cls, duration_ms: int, sample_rate: int = DEFAULT_SAMPLE_RATE
    ) -> "PcmClip":
        count = int(sample_rate * duration_ms / 1000)
        return cls(data=b"\x00" * (count * BYTES_PER_SAMPLE), sample_rate=sample_rate)

    def __add__(self, other: "PcmClip") -> "PcmClip":
        if self.sample_rate != other.sample_rate:
            raise AudioError(
                f"cannot concatenate {self.sample_rate} Hz with "
                f"{other.sample_rate} Hz audio"
            )
        return PcmClip(self.data + other.data, self.sample_rate)


def rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of a PCM buffer, normalised to 0.0-1.0.

    Pure Python on purpose. At a 20 ms frame this is under a thousand samples
    per call and a few thousand calls per session -- far below the point where
    pulling in numpy would buy anything, and numpy is a large dependency to
    carry for one arithmetic mean.
    """
    samples = _as_samples(pcm)
    if not samples:
        return 0.0
    total = 0
    for sample in samples:
        total += sample * sample
    return math.sqrt(total / len(samples)) / INT16_MAX


def peak(pcm: bytes) -> float:
    """Peak absolute amplitude, normalised to 0.0-1.0."""
    samples = _as_samples(pcm)
    if not samples:
        return 0.0
    return max(abs(s) for s in samples) / INT16_MAX


def _as_samples(pcm: bytes) -> array.array:
    """Reinterpret a byte buffer as int16 samples, ignoring a trailing odd byte."""
    if not pcm:
        return array.array("h")
    usable = len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)
    samples = array.array("h")
    samples.frombytes(pcm[:usable])
    return samples


def estimate_duration_ms(text: str, speech_rate: float = 1.0) -> int:
    """Rough spoken duration of a piece of text.

    Used only by the offline backends, where no real synthesiser is available
    but the *length* of an utterance still has to be realistic -- a loop tested
    against instantaneous audio would not exercise the turn-taking logic at all.

    Fourteen characters per second is about 150 words per minute of English
    prose, which is unremarkable conversational speed.
    """
    if speech_rate <= 0:
        raise AudioError(f"speech_rate must be positive, got {speech_rate}")
    chars_per_second = 14.0 * speech_rate
    seconds = max(len(text.strip()), 1) / chars_per_second
    return int(max(seconds, 0.25) * 1000)


def synthesize_speech_like(
    duration_ms: int,
    seed: int = 0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    amplitude: float = 0.35,
) -> PcmClip:
    """Generate audio that is not speech but behaves like it for the loop.

    A voiced fundamental with two harmonics, amplitude-modulated at roughly
    syllable rate and gated by short inter-word pauses. It is not intelligible
    and is not meant to be: its job is to have the right *envelope*, so an
    energy detector sees speech start, speech stop, and gaps in between exactly
    where a real utterance would have them.

    That is what lets the entire week 2 loop -- turn taking, barge-in, dead-air
    accounting -- be exercised and tested with no provider, no key and no
    network. Deterministic in ``seed`` so a failing run can be reproduced.
    """
    if duration_ms <= 0:
        raise AudioError(f"duration_ms must be positive, got {duration_ms}")
    if not 0.0 < amplitude <= 1.0:
        raise AudioError(f"amplitude must be in (0.0, 1.0], got {amplitude}")

    count = int(sample_rate * duration_ms / 1000)

    # A small deterministic PRNG rather than `random`, so this never depends on
    # global interpreter random state that a caller might have reseeded.
    state = (seed * 1_103_515_245 + 12_345) & 0x7FFF_FFFF

    def next_unit() -> float:
        nonlocal state
        state = (state * 1_103_515_245 + 12_345) & 0x7FFF_FFFF
        return state / 0x7FFF_FFFF

    syllable_hz = 3.2 + next_unit() * 1.6     # syllables per second
    pause_every = 6 + int(next_unit() * 5)    # syllables between word gaps

    # The carrier repeats every fundamental period, so it is generated once and
    # tiled rather than recomputed per sample. Rounding the period to a whole
    # number of samples (and deriving the pitch back from it) keeps the tiling
    # phase-continuous, so no discontinuity is introduced at the seam.
    #
    # Naively this function ran three sin() calls per sample: for a four-second
    # utterance at 48 kHz that is well over half a million, and it dominated the
    # entire test suite. A generator for a signal nobody listens to is not worth
    # a second of every run.
    period = max(2, round(sample_rate / (95.0 + next_unit() * 85.0)))
    carrier = [
        (
            math.sin(2 * math.pi * i / period)
            + 0.5 * math.sin(4 * math.pi * i / period)
            + 0.25 * math.sin(6 * math.pi * i / period)
        )
        / 1.75
        for i in range(period)
    ]

    # The envelope moves at a few hertz, so holding it constant across a short
    # block is inaudible and indistinguishable to any energy detector, while
    # cutting the trigonometry by a further two orders of magnitude.
    block = max(1, sample_rate // 1_500)

    samples = array.array("h", bytes(count * BYTES_PER_SAMPLE))
    scale = amplitude * INT16_MAX
    carrier_index = 0
    for start in range(0, count, block):
        syllable_phase = (start / sample_rate) * syllable_hz
        # Raised cosine per syllable: energy rises and falls the way a spoken
        # syllable does, rather than switching on and off like a tone burst.
        envelope = 0.5 - 0.5 * math.cos(2 * math.pi * (syllable_phase % 1.0))
        if int(syllable_phase) % pause_every == pause_every - 1:
            envelope *= 0.04  # inter-word gap: near silence, not digital zero
        gain = envelope * scale
        for i in range(start, min(start + block, count)):
            value = carrier[carrier_index] * gain
            carrier_index += 1
            if carrier_index == period:
                carrier_index = 0
            samples[i] = (
                INT16_MAX
                if value > INT16_MAX
                else -INT16_MAX
                if value < -INT16_MAX
                else int(value)
            )

    return PcmClip(samples.tobytes(), sample_rate=sample_rate)


def concat(clips: List[PcmClip]) -> PcmClip:
    """Join clips, erroring rather than silently mixing sample rates."""
    if not clips:
        raise AudioError("cannot concatenate an empty list of clips")
    rate = clips[0].sample_rate
    for clip in clips:
        if clip.sample_rate != rate:
            raise AudioError(
                f"mixed sample rates in concat: {rate} Hz and {clip.sample_rate} Hz"
            )
    return PcmClip(b"".join(c.data for c in clips), sample_rate=rate)


def wav_header_bytes(sample_rate: int = DEFAULT_SAMPLE_RATE, channels: int = 1) -> bytes:
    """A streaming WAV header with placeholder lengths, patched on close."""
    byte_rate = sample_rate * channels * BYTES_PER_SAMPLE
    return (
        b"RIFF"
        + struct.pack("<I", 0)
        + b"WAVE"
        + b"fmt "
        + struct.pack(
            "<IHHIIHH",
            16,                                # fmt chunk size
            1,                                 # PCM
            channels,
            sample_rate,
            byte_rate,
            channels * BYTES_PER_SAMPLE,       # block align
            16,                                # bits per sample
        )
        + b"data"
        + struct.pack("<I", 0)
    )


class StreamingWavWriter:
    """Append PCM frames to a WAV file as they arrive, patching the header on close.

    Buffering a whole session in memory and writing it at the end would work
    right up until a session hangs -- which is precisely the run worth having a
    recording of. Writing on arrival means a killed run still leaves usable
    audio on disk.
    """

    def __init__(self, path: Path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.bytes_written = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(path, "wb")
        self._handle.write(wav_header_bytes(sample_rate))

    def write(self, pcm: bytes) -> None:
        if self._handle.closed:
            raise AudioError(f"{self.path}: write after close")
        self._handle.write(pcm)
        self.bytes_written += len(pcm)

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.seek(4)
        self._handle.write(struct.pack("<I", 36 + self.bytes_written))
        self._handle.seek(40)
        self._handle.write(struct.pack("<I", self.bytes_written))
        self._handle.close()

    @property
    def duration_ms(self) -> int:
        samples = self.bytes_written // BYTES_PER_SAMPLE
        return int(round(samples * 1000 / self.sample_rate))

    def __enter__(self) -> "StreamingWavWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
