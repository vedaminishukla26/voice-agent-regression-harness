"""Persona specifications for simulated speakers.

A ``PersonaSpec`` describes *how* a simulated candidate speaks, independent of
*what* they say (that lives in :mod:`candidate_scripts`). Keeping the two apart
means any script can be rendered through any persona, which is the point of the
harness: the same words at a different speech rate, or with a different
interruption profile, exercise completely different code paths in the agent
under test.

Every field here is a knob that plausibly changes agent behaviour.
``silence_before_reply_ms`` in particular interacts directly with
voice-activity-detection endpointing thresholds, which are typically a single
global constant in an agent implementation and therefore mistuned for at least
some of the languages that agent serves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

# Bounds enforced in __post_init__. Deliberately generous: the point of the
# harness is to probe edges, so a persona is allowed to be extreme -- just not
# incoherent.
MIN_SPEECH_RATE = 0.5
MAX_SPEECH_RATE = 2.0
MAX_SILENCE_BEFORE_REPLY_MS = 30_000


@dataclass(frozen=True)
class PersonaSpec:
    """How a simulated speaker sounds and behaves in a conversation."""

    name: str
    description: str

    # Primary language as a short code ("en", "hi", "es"). Passed to the TTS
    # backend and used to group results when comparing per-language behaviour.
    language: str = "en"

    # Free-form accent hint. Backends that cannot honour it ignore it.
    accent: Optional[str] = None

    # Multiplier on natural speaking rate. 1.0 is the voice's default.
    speech_rate: float = 1.0

    # How long the speaker stays silent before answering. Long pauses are the
    # most direct way to provoke premature endpointing.
    silence_before_reply_ms: int = 0

    # 0.0 never interrupts; 1.0 talks over the agent at every opportunity.
    interruption_aggression: float = 0.0

    # Proportion of turns carrying a disfluency ("um", "so, like").
    filler_rate: float = 0.0

    # Secondary language for mid-utterance code switching. None disables it.
    # Multilingual STT handles switching unevenly, and the failure mode is
    # silent: a plausible-looking but wrong transcript.
    code_switch_to: Optional[str] = None

    # Backend-specific voice identifier. None lets the backend choose.
    voice_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("persona name must be a non-empty string")
        if not self.description or not self.description.strip():
            raise ValueError(f"persona {self.name!r}: description must not be empty")
        if not self.language or not self.language.strip():
            raise ValueError(f"persona {self.name!r}: language must not be empty")

        if not MIN_SPEECH_RATE <= self.speech_rate <= MAX_SPEECH_RATE:
            raise ValueError(
                f"persona {self.name!r}: speech_rate {self.speech_rate} outside "
                f"[{MIN_SPEECH_RATE}, {MAX_SPEECH_RATE}]"
            )
        if not 0 <= self.silence_before_reply_ms <= MAX_SILENCE_BEFORE_REPLY_MS:
            raise ValueError(
                f"persona {self.name!r}: silence_before_reply_ms "
                f"{self.silence_before_reply_ms} outside "
                f"[0, {MAX_SILENCE_BEFORE_REPLY_MS}]"
            )
        if not 0.0 <= self.interruption_aggression <= 1.0:
            raise ValueError(
                f"persona {self.name!r}: interruption_aggression must be in [0.0, 1.0]"
            )
        if not 0.0 <= self.filler_rate <= 1.0:
            raise ValueError(f"persona {self.name!r}: filler_rate must be in [0.0, 1.0]")
        if self.code_switch_to is not None and self.code_switch_to == self.language:
            raise ValueError(
                f"persona {self.name!r}: code_switch_to must differ from language"
            )

    def to_dict(self) -> Dict:
        """Serialisable form, for embedding in result artifacts."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Predefined personas
# ---------------------------------------------------------------------------
# Each targets a specific class of agent failure. A persona that does not
# plausibly break something does not belong here.

COOPERATIVE = PersonaSpec(
    name="cooperative",
    description="Answers directly and at a normal pace. The control case.",
    language="en",
    accent="neutral",
    speech_rate=1.0,
    silence_before_reply_ms=300,
)

NERVOUS = PersonaSpec(
    name="nervous",
    description=(
        "Hesitant and vague, with long pauses mid-answer. Probes whether the "
        "agent waits for a complete thought or cuts in at the first gap."
    ),
    language="en",
    accent="neutral",
    speech_rate=0.85,
    silence_before_reply_ms=2_200,
    filler_rate=0.6,
)

DIFFICULT = PersonaSpec(
    name="difficult",
    description=(
        "Deflects, asks to skip questions, and pushes back. Probes whether the "
        "agent re-asks unanswered questions instead of moving on."
    ),
    language="en",
    speech_rate=1.1,
    silence_before_reply_ms=200,
    interruption_aggression=0.4,
)

EDGE_CASE = PersonaSpec(
    name="edge_case",
    description=(
        "Very short answers, frequent interruptions, and long silences. Probes "
        "endpointing and barge-in handling harder than any other persona."
    ),
    language="en",
    speech_rate=1.3,
    silence_before_reply_ms=4_500,
    interruption_aggression=0.9,
)

ADVERSARIAL = PersonaSpec(
    name="adversarial",
    description=(
        "Attempts to break character, extract the system prompt, and steer the "
        "agent off task. Probes guardrail integrity through the audio path, "
        "where instructions arrive as transcribed speech rather than as text."
    ),
    language="en",
    speech_rate=1.0,
    silence_before_reply_ms=250,
    interruption_aggression=0.2,
)

HINDI_MULTILINGUAL = PersonaSpec(
    name="hindi_multilingual",
    description=(
        "Switches between Hindi and English mid-sentence. Probes multilingual "
        "STT and any endpointing threshold tuned only against English."
    ),
    language="hi",
    accent="indian",
    speech_rate=1.0,
    silence_before_reply_ms=600,
    filler_rate=0.2,
    code_switch_to="en",
)

SPANISH_ACCENT = PersonaSpec(
    name="spanish_accent",
    description=(
        "English spoken with a strong Spanish accent. Probes transcription "
        "where the language is correct but the phonetics sit outside the "
        "model's dominant training distribution."
    ),
    language="en",
    accent="spanish",
    speech_rate=0.95,
    silence_before_reply_ms=500,
)


PERSONAS: Dict[str, PersonaSpec] = {
    p.name: p
    for p in (
        COOPERATIVE,
        NERVOUS,
        DIFFICULT,
        EDGE_CASE,
        ADVERSARIAL,
        HINDI_MULTILINGUAL,
        SPANISH_ACCENT,
    )
}


def get_persona(name: str) -> PersonaSpec:
    """Look up a persona by name, erroring with the list of valid names."""
    try:
        return PERSONAS[name]
    except KeyError:
        raise KeyError(
            f"unknown persona {name!r}; available: {', '.join(sorted(PERSONAS))}"
        ) from None


def _main() -> None:
    """Print the persona table. Smoke check only: no network, no keys."""
    header = f"{'NAME':<20} {'LANG':<6} {'RATE':<6} {'PAUSE':<8} {'INTERRUPT':<10}"
    print(header)
    print("-" * len(header))
    for name in sorted(PERSONAS):
        p = PERSONAS[name]
        print(
            f"{p.name:<20} {p.language:<6} {p.speech_rate:<6.2f} "
            f"{p.silence_before_reply_ms:<8} {p.interruption_aggression:<10.2f}"
        )
    print(f"\n{len(PERSONAS)} personas defined.")


if __name__ == "__main__":
    _main()
