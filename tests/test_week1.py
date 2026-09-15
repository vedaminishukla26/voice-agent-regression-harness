"""Week 1 validation: personas, scripts, and TTS plumbing.

Every test here runs offline. No API key, no network, no audio. That is a
requirement, not a convenience: a test suite that needs an account is a test
suite that stops being run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from candidate_scripts import (
    INTENTS,
    SCRIPTS,
    CandidateScript,
    CandidateTurn,
    get_script,
)
from persona_spec import (
    MAX_SILENCE_BEFORE_REPLY_MS,
    MAX_SPEECH_RATE,
    MIN_SPEECH_RATE,
    PERSONAS,
    PersonaSpec,
    get_persona,
)
from persona_tts import (
    CartesiaTTSBackend,
    DryRunBackend,
    RenderResult,
    TTSError,
    build_parser,
    main,
    output_path_for,
    render_persona_batch,
    render_turn,
    slugify,
    write_results,
)


# ---------------------------------------------------------------------------
# PersonaSpec
# ---------------------------------------------------------------------------


def test_persona_registry_is_populated():
    assert len(PERSONAS) == 7


def test_every_persona_is_keyed_by_its_own_name():
    for key, persona in PERSONAS.items():
        assert key == persona.name


def test_expected_personas_are_present():
    expected = {
        "cooperative",
        "nervous",
        "difficult",
        "edge_case",
        "adversarial",
        "hindi_multilingual",
        "spanish_accent",
    }
    assert set(PERSONAS) == expected


def test_personas_are_immutable():
    with pytest.raises(Exception):
        PERSONAS["cooperative"].speech_rate = 1.5


def test_get_persona_returns_the_right_one():
    assert get_persona("nervous").name == "nervous"


def test_get_persona_rejects_unknown_name_and_lists_valid_ones():
    with pytest.raises(KeyError) as exc:
        get_persona("does_not_exist")
    assert "cooperative" in str(exc.value)


def test_persona_rejects_empty_name():
    with pytest.raises(ValueError):
        PersonaSpec(name="  ", description="x")


def test_persona_rejects_empty_description():
    with pytest.raises(ValueError):
        PersonaSpec(name="x", description="")


def test_persona_rejects_speech_rate_below_bound():
    with pytest.raises(ValueError, match="speech_rate"):
        PersonaSpec(name="x", description="d", speech_rate=MIN_SPEECH_RATE - 0.01)


def test_persona_rejects_speech_rate_above_bound():
    with pytest.raises(ValueError, match="speech_rate"):
        PersonaSpec(name="x", description="d", speech_rate=MAX_SPEECH_RATE + 0.01)


def test_persona_rejects_negative_silence():
    with pytest.raises(ValueError, match="silence_before_reply_ms"):
        PersonaSpec(name="x", description="d", silence_before_reply_ms=-1)


def test_persona_rejects_absurd_silence():
    with pytest.raises(ValueError, match="silence_before_reply_ms"):
        PersonaSpec(
            name="x", description="d", silence_before_reply_ms=MAX_SILENCE_BEFORE_REPLY_MS + 1
        )


def test_persona_rejects_out_of_range_interruption():
    with pytest.raises(ValueError, match="interruption_aggression"):
        PersonaSpec(name="x", description="d", interruption_aggression=1.5)


def test_persona_rejects_out_of_range_filler_rate():
    with pytest.raises(ValueError, match="filler_rate"):
        PersonaSpec(name="x", description="d", filler_rate=-0.1)


def test_persona_rejects_code_switch_to_same_language():
    with pytest.raises(ValueError, match="code_switch_to"):
        PersonaSpec(name="x", description="d", language="en", code_switch_to="en")


def test_persona_serialises_to_dict():
    data = PERSONAS["cooperative"].to_dict()
    assert data["name"] == "cooperative"
    assert "speech_rate" in data
    json.dumps(data)  # must be JSON-safe for result artifacts


def test_all_personas_are_within_declared_bounds():
    for persona in PERSONAS.values():
        assert MIN_SPEECH_RATE <= persona.speech_rate <= MAX_SPEECH_RATE
        assert 0 <= persona.silence_before_reply_ms <= MAX_SILENCE_BEFORE_REPLY_MS
        assert 0.0 <= persona.interruption_aggression <= 1.0
        assert 0.0 <= persona.filler_rate <= 1.0


def test_personas_cover_more_than_one_language():
    # A harness that only ever speaks English cannot find a language-specific bug.
    assert len({p.language for p in PERSONAS.values()}) > 1


def test_at_least_one_persona_interrupts_aggressively():
    assert any(p.interruption_aggression >= 0.8 for p in PERSONAS.values())


def test_at_least_one_persona_pauses_long_enough_to_trip_endpointing():
    assert any(p.silence_before_reply_ms >= 2_000 for p in PERSONAS.values())


# ---------------------------------------------------------------------------
# CandidateScript
# ---------------------------------------------------------------------------


def test_script_registry_is_populated():
    assert len(SCRIPTS) == 7


def test_every_persona_has_a_script():
    for name in PERSONAS:
        assert name in SCRIPTS, f"no script for persona {name}"


def test_no_script_is_orphaned():
    for name in SCRIPTS:
        assert name in PERSONAS, f"script {name} has no matching persona"


def test_every_script_is_keyed_by_its_persona():
    for key, script in SCRIPTS.items():
        assert key == script.persona


def test_every_script_has_turns():
    for name, script in SCRIPTS.items():
        assert len(script) > 0, f"script {name} is empty"


def test_every_turn_has_non_empty_text():
    for script in SCRIPTS.values():
        for turn in script.turns:
            assert turn.text.strip()


def test_every_turn_uses_a_known_intent():
    for script in SCRIPTS.values():
        for turn in script.turns:
            assert turn.intent in INTENTS


def test_get_script_rejects_unknown_persona():
    with pytest.raises(KeyError):
        get_script("nobody")


def test_turn_rejects_empty_text():
    with pytest.raises(ValueError):
        CandidateTurn(text="   ", intent="answering")


def test_turn_rejects_unknown_intent():
    with pytest.raises(ValueError, match="unknown intent"):
        CandidateTurn(text="hello", intent="vibing")


def test_turn_rejects_negative_delay():
    with pytest.raises(ValueError, match="delay_ms"):
        CandidateTurn(text="hello", intent="answering", delay_ms=-5)


def test_script_rejects_empty_turn_list():
    with pytest.raises(ValueError, match="no turns"):
        CandidateScript(persona="x", turns=[])


def test_by_intent_filters_correctly():
    script = get_script("adversarial")
    injecting = script.by_intent("injecting")
    assert injecting
    assert all(t.intent == "injecting" for t in injecting)


def test_by_intent_returns_empty_for_absent_intent():
    assert get_script("cooperative").by_intent("injecting") == []


def test_adversarial_script_actually_attempts_injection():
    assert "injecting" in get_script("adversarial").intents()


def test_difficult_script_actually_deflects():
    assert "deflecting" in get_script("difficult").intents()


def test_edge_case_script_has_a_very_long_pause():
    delays = [t.delay_ms for t in get_script("edge_case").turns if t.delay_ms]
    assert delays and max(delays) >= 8_000


def test_script_serialises_to_json():
    json.dumps(get_script("cooperative").to_dict())


# ---------------------------------------------------------------------------
# TTS plumbing
# ---------------------------------------------------------------------------


def test_dry_run_backend_makes_no_network_call_and_returns_bytes():
    backend = DryRunBackend()
    out = backend.synthesize("hello", PERSONAS["cooperative"])
    assert isinstance(out, bytes) and out
    assert backend.calls == ["hello"]


def test_slugify_is_filename_safe():
    assert slugify("Hello, world! 123") == "hello_world_123"


def test_slugify_handles_unprintable_input():
    assert slugify("!!!???") == "turn"


def test_slugify_respects_limit():
    assert len(slugify("a" * 200, limit=10)) == 10


def test_output_path_is_sortable_and_descriptive():
    turn = get_script("cooperative").turns[0]
    path = output_path_for(PERSONAS["cooperative"], 3, turn, Path("/tmp"), "mp3")
    assert path.name.startswith("cooperative_03_answering_")
    assert path.suffix == ".mp3"


def test_render_turn_writes_a_file(tmp_path):
    turn = get_script("cooperative").turns[0]
    result = render_turn(DryRunBackend(), PERSONAS["cooperative"], turn, 1, tmp_path)
    assert result.ok
    assert Path(result.path).exists()
    assert result.bytes_written > 0


def test_render_turn_records_failure_instead_of_raising(tmp_path):
    class Broken:
        extension = "mp3"

        def synthesize(self, text, persona):
            raise TTSError("provider exploded")

    turn = get_script("cooperative").turns[0]
    result = render_turn(Broken(), PERSONAS["cooperative"], turn, 1, tmp_path)
    assert not result.ok
    assert "provider exploded" in result.error
    assert result.path is None


def test_render_persona_batch_covers_every_turn(tmp_path):
    results = render_persona_batch(
        "cooperative", DryRunBackend(), tmp_path, verbose=False
    )
    assert len(results) == len(get_script("cooperative"))
    assert all(r.ok for r in results)


def test_write_results_produces_valid_json(tmp_path):
    results = render_persona_batch("nervous", DryRunBackend(), tmp_path, verbose=False)
    path = write_results(results, tmp_path, "nervous")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["total"] == len(results)
    assert payload["failed"] == 0


def test_write_results_survives_non_ascii(tmp_path):
    results = render_persona_batch(
        "hindi_multilingual", DryRunBackend(), tmp_path, verbose=False
    )
    path = write_results(results, tmp_path, "hindi")
    json.loads(path.read_text(encoding="utf-8"))


def test_render_result_serialises():
    json.dumps(RenderResult("p", 1, "answering", "t", True).to_dict())


def test_cartesia_backend_refuses_to_construct_without_key():
    with pytest.raises(TTSError, match="no API key"):
        CartesiaTTSBackend(api_key="", voice_id="v")


def test_cartesia_backend_refuses_to_construct_without_voice():
    with pytest.raises(TTSError, match="no voice id"):
        CartesiaTTSBackend(api_key="k", voice_id="")


def test_cartesia_payload_carries_persona_delivery_settings():
    backend = CartesiaTTSBackend(api_key="k", voice_id="default-voice")
    payload = backend.build_payload("hello", PERSONAS["hindi_multilingual"])
    assert payload["transcript"] == "hello"
    assert payload["language"] == "hi"
    assert payload["speed"] == PERSONAS["hindi_multilingual"].speech_rate
    assert payload["output_format"]["container"] == "mp3"


def test_cartesia_payload_prefers_persona_voice_over_default():
    backend = CartesiaTTSBackend(api_key="k", voice_id="default-voice")
    persona = PersonaSpec(name="x", description="d", voice_id="persona-voice")
    assert backend.build_payload("hi", persona)["voice"]["id"] == "persona-voice"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_parser_rejects_persona_and_all_personas_together():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--persona", "cooperative", "--all-personas"])


def test_parser_rejects_unknown_persona():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--persona", "nobody"])


def test_cli_dry_run_succeeds_without_any_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    code = main(["--all-personas", "--dry-run", "--output-dir", str(tmp_path), "--quiet"])
    assert code == 0


def test_cli_refuses_to_run_live_without_a_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    code = main(["--persona", "cooperative", "--output-dir", str(tmp_path)])
    assert code == 2


def test_cli_defaults_to_dry_run_when_given_no_target(tmp_path, monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    assert main(["--output-dir", str(tmp_path), "--quiet"]) == 0
