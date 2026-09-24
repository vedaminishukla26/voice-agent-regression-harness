"""Week 2 tests: PCM handling, speech detection, session recording, the loop.

Everything here is offline. No server, no API key, no network, no sleeping
through real conversation time -- sessions run against the loopback transport
under the virtual clock, so a three-minute conversation costs milliseconds.

Async tests call ``asyncio.run`` directly rather than pulling in an async pytest
plugin. One test runner remains the only dependency this project has.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from audio_io import (
    BYTES_PER_SAMPLE,
    DEFAULT_SAMPLE_RATE,
    AudioError,
    PcmClip,
    StreamingWavWriter,
    concat,
    estimate_duration_ms,
    peak,
    rms,
    synthesize_speech_like,
)
from audio_loop import ClipSource, SessionConfig, TurnSelector, run_session
from candidate_scripts import CandidateScript, CandidateTurn, get_script
from persona_spec import get_persona
from session_log import (
    ACTOR_AGENT,
    ACTOR_CANDIDATE,
    ACTOR_HARNESS,
    STOP_AGENT_NEVER_JOINED,
    STOP_AGENT_SILENT,
    STOP_SCRIPT_EXHAUSTED,
    STOP_TURN_CEILING,
    STOP_WALL_CLOCK,
    FakeClock,
    SessionEvent,
    SessionLog,
    VirtualScheduler,
    load_session,
    race_timeout,
)
from transport import LoopbackAgent, LoopbackTransport
from vad import SPEECH_END, SPEECH_START, SpeechDetector, VadConfig

# ---------------------------------------------------------------------------
# audio_io
# ---------------------------------------------------------------------------


def test_clip_rejects_odd_length_buffer():
    with pytest.raises(AudioError, match="whole number of 16-bit samples"):
        PcmClip(b"\x00\x00\x00")


def test_clip_rejects_stereo():
    with pytest.raises(AudioError, match="only mono"):
        PcmClip(b"\x00\x00", channels=2)


def test_clip_rejects_non_positive_sample_rate():
    with pytest.raises(AudioError):
        PcmClip(b"\x00\x00", sample_rate=0)


def test_duration_matches_sample_count():
    clip = PcmClip(b"\x00\x00" * DEFAULT_SAMPLE_RATE)
    assert clip.duration_ms == 1000
    assert clip.sample_count == DEFAULT_SAMPLE_RATE


def test_frames_are_all_full_size_and_last_is_padded():
    # One and a half frames of audio must still yield two whole frames.
    per_frame = int(DEFAULT_SAMPLE_RATE * 20 / 1000)
    clip = PcmClip(b"\x11\x11" * (per_frame + per_frame // 2))
    frames = list(clip.frames(20))
    assert len(frames) == 2
    assert {len(f) for f in frames} == {per_frame * BYTES_PER_SAMPLE}
    assert frames[1].endswith(b"\x00" * 16)


def test_frame_count_of_empty_clip_is_zero():
    assert PcmClip(b"").frame_count() == 0


def test_silence_has_zero_energy_and_speech_does_not():
    assert rms(PcmClip.silence(100).data) == 0.0
    assert peak(PcmClip.silence(100).data) == 0.0
    speech = synthesize_speech_like(400, seed=1)
    assert rms(speech.data) > 0.02
    assert peak(speech.data) > 0.1


def test_rms_ignores_a_trailing_odd_byte():
    assert rms(b"\x00\x00\x7f") == pytest.approx(0.0)


def test_synthesize_is_deterministic_in_seed():
    assert synthesize_speech_like(200, seed=5).data == synthesize_speech_like(200, seed=5).data
    assert synthesize_speech_like(200, seed=5).data != synthesize_speech_like(200, seed=6).data


def test_synthesize_rejects_bad_arguments():
    with pytest.raises(AudioError):
        synthesize_speech_like(0)
    with pytest.raises(AudioError):
        synthesize_speech_like(100, amplitude=0.0)


def test_estimate_duration_scales_with_speech_rate():
    slow = estimate_duration_ms("a" * 100, speech_rate=0.5)
    fast = estimate_duration_ms("a" * 100, speech_rate=2.0)
    assert slow > fast
    with pytest.raises(AudioError):
        estimate_duration_ms("hello", speech_rate=0)


def test_concat_rejects_mixed_sample_rates():
    with pytest.raises(AudioError, match="mixed sample rates"):
        concat([PcmClip(b"\x00\x00", 48_000), PcmClip(b"\x00\x00", 44_100)])
    with pytest.raises(AudioError):
        concat([])


def test_wav_roundtrip_preserves_audio(tmp_path: Path):
    clip = synthesize_speech_like(300, seed=2)
    clip.to_wav(tmp_path / "a.wav")
    assert PcmClip.from_wav(tmp_path / "a.wav").data == clip.data


def test_streaming_writer_patches_header_so_the_file_is_readable(tmp_path: Path):
    clip = synthesize_speech_like(200, seed=3)
    writer = StreamingWavWriter(tmp_path / "s.wav")
    for frame in clip.frames(20):
        writer.write(frame)
    writer.close()
    assert PcmClip.from_wav(tmp_path / "s.wav").duration_ms == writer.duration_ms


def test_streaming_writer_rejects_write_after_close(tmp_path: Path):
    writer = StreamingWavWriter(tmp_path / "s.wav")
    writer.close()
    with pytest.raises(AudioError, match="write after close"):
        writer.write(b"\x00\x00")


# ---------------------------------------------------------------------------
# vad
# ---------------------------------------------------------------------------


def _frames_of(clip: PcmClip, frame_ms: int = 20):
    return list(clip.frames(frame_ms))


def test_vad_config_rejects_thresholds_shorter_than_a_frame():
    with pytest.raises(ValueError, match="shorter than one frame"):
        VadConfig(start_ms=10, frame_ms=20)
    with pytest.raises(ValueError, match="threshold_rms"):
        VadConfig(threshold_rms=0.0)


def test_detector_reports_speech_start_retroactively():
    """The start must be the frame speech began, not the frame it was proven."""
    detector = SpeechDetector(VadConfig(start_ms=60, frame_ms=20))
    silence = PcmClip.silence(20).data
    loud = _frames_of(synthesize_speech_like(400, seed=9))[5]

    events = []
    t = 0
    for _ in range(5):
        events += detector.push(silence, t)
        t += 20
    speech_began_at = t
    for _ in range(3):
        events += detector.push(loud, t)
        t += 20

    starts = [e for e in events if e.kind == SPEECH_START]
    assert len(starts) == 1
    assert starts[0].t_ms == speech_began_at
    assert starts[0].declared_at_ms > starts[0].t_ms
    assert starts[0].detection_lag_ms == 40


def test_detector_does_not_split_an_utterance_at_a_short_gap():
    """A pause shorter than end_ms is within one utterance, not between two."""
    detector = SpeechDetector(VadConfig(start_ms=40, end_ms=400, frame_ms=20))
    loud = _frames_of(synthesize_speech_like(400, seed=4))[5]
    quiet = PcmClip.silence(20).data

    events = []
    t = 0
    for _ in range(10):
        events += detector.push(loud, t)
        t += 20
    for _ in range(5):        # 100 ms gap, well under end_ms
        events += detector.push(quiet, t)
        t += 20
    for _ in range(10):
        events += detector.push(loud, t)
        t += 20

    assert [e.kind for e in events] == [SPEECH_START]
    assert detector.is_speaking


def test_detector_reports_speech_end_at_the_start_of_the_silence():
    detector = SpeechDetector(VadConfig(start_ms=40, end_ms=100, frame_ms=20))
    loud = _frames_of(synthesize_speech_like(400, seed=4))[5]
    quiet = PcmClip.silence(20).data

    t = 0
    for _ in range(10):
        detector.push(loud, t)
        t += 20
    silence_began_at = t
    events = []
    for _ in range(6):
        events += detector.push(quiet, t)
        t += 20

    ends = [e for e in events if e.kind == SPEECH_END]
    assert len(ends) == 1
    assert ends[0].t_ms == silence_began_at
    assert not detector.is_speaking


def test_flush_closes_an_utterance_left_open():
    detector = SpeechDetector(VadConfig(start_ms=40, frame_ms=20))
    loud = _frames_of(synthesize_speech_like(400, seed=4))[5]
    for t in range(0, 200, 20):
        detector.push(loud, t)
    assert detector.is_speaking
    events = detector.flush(999)
    assert [e.kind for e in events] == [SPEECH_END]
    assert events[0].t_ms == 999
    assert detector.flush(1000) == []


def test_detector_accounting_adds_up():
    detector = SpeechDetector(VadConfig(frame_ms=20))
    for t in range(0, 200, 20):
        detector.push(PcmClip.silence(20).data, t)
    summary = detector.to_dict()
    assert summary["frames_seen"] == 10
    assert summary["observed_ms"] == summary["speech_ms"] + summary["silence_ms"]


# ---------------------------------------------------------------------------
# session_log
# ---------------------------------------------------------------------------


def test_event_rejects_negative_time_and_unknown_actor():
    with pytest.raises(ValueError):
        SessionEvent(t_ms=-1, kind="x", actor=ACTOR_HARNESS)
    with pytest.raises(ValueError, match="unknown actor"):
        SessionEvent(t_ms=0, kind="x", actor="nobody")


def test_retroactive_event_is_kept_in_time_order():
    log = SessionLog("cooperative", "room", clock=FakeClock(0))
    log.clock.set(1000)
    log.record("late", ACTOR_AGENT)
    log.record("early", ACTOR_AGENT, t_ms=500)
    assert [e.t_ms for e in log.events] == sorted(e.t_ms for e in log.events)
    assert [e.kind for e in log.events] == ["session_start", "early", "late"]


def test_end_turn_closes_the_most_recent_open_turn_for_that_actor():
    log = SessionLog("cooperative", "room", clock=FakeClock(0))
    log.begin_turn(ACTOR_AGENT, 0)
    log.begin_turn(ACTOR_CANDIDATE, 100)
    log.end_turn(ACTOR_AGENT, 50)
    assert log.turns[0].end_ms == 50
    assert log.turns[1].end_ms is None
    assert log.open_turn(ACTOR_CANDIDATE) is log.turns[1]
    assert log.end_turn(ACTOR_AGENT, 60) is None


def test_finish_is_idempotent_so_the_first_reason_wins():
    log = SessionLog("cooperative", "room", clock=FakeClock(0))
    log.finish(STOP_WALL_CLOCK)
    log.finish(STOP_SCRIPT_EXHAUSTED)
    assert log.stop_reason == STOP_WALL_CLOCK


def test_resolve_overlaps_confirms_only_real_overlap():
    """An interruption that missed is not a barge-in, however it was intended."""
    log = SessionLog("edge_case", "room", clock=FakeClock(0))
    log.begin_turn(ACTOR_AGENT, 0)
    log.end_turn(ACTOR_AGENT, 1000)

    hit = log.begin_turn(ACTOR_CANDIDATE, 800, intended_barge_in=True)
    hit.end_ms = 1500
    missed = log.begin_turn(ACTOR_CANDIDATE, 2000, intended_barge_in=True)
    missed.end_ms = 2500

    log.resolve_overlaps()
    assert hit.overlap_ms == 200 and hit.barge_in is True
    assert missed.overlap_ms == 0 and missed.barge_in is False
    counts = log.to_dict()["counts"]
    assert counts["intended_barge_ins"] == 2
    assert counts["barge_ins"] == 1
    assert counts["overlap_ms"] == 200


def test_resolve_overlaps_ignores_unclosed_turns():
    log = SessionLog("edge_case", "room", clock=FakeClock(0))
    log.begin_turn(ACTOR_AGENT, 0)
    open_turn = log.begin_turn(ACTOR_CANDIDATE, 100, intended_barge_in=True)
    log.resolve_overlaps()
    assert open_turn.overlap_ms is None


def test_session_round_trips_through_disk(tmp_path: Path):
    log = SessionLog("cooperative", "room", clock=FakeClock(0), config={"k": 1})
    log.record("thing", ACTOR_AGENT, t_ms=10, detail="x")
    log.finish(STOP_SCRIPT_EXHAUSTED)
    path = log.write(tmp_path)
    back = load_session(path)
    assert back["persona"] == "cooperative"
    assert back["stop_reason"] == STOP_SCRIPT_EXHAUSTED
    assert back["config"]["k"] == 1
    assert any(e["kind"] == "thing" for e in back["events"])
    assert json.loads(path.read_text(encoding="utf-8")) == back


def test_fake_clock_refuses_to_run_backwards():
    clock = FakeClock(0)
    clock.advance(10)
    with pytest.raises(ValueError):
        clock.advance(-1)


# ---------------------------------------------------------------------------
# virtual scheduler
# ---------------------------------------------------------------------------


def test_virtual_clock_sleeps_are_exact_and_ordered():
    """Simulated sleeps must be exact; compressed real sleeps never are."""

    async def scenario():
        scheduler = VirtualScheduler()
        scheduler.start()
        order = []

        async def sleeper(name, ms):
            await scheduler.sleep_ms(ms)
            order.append((name, scheduler.now_ms()))

        await asyncio.gather(
            sleeper("late", 5_000), sleeper("early", 20), sleeper("mid", 1_000)
        )
        await scheduler.stop()
        return order

    order = asyncio.run(scenario())
    assert order == [("early", 20), ("mid", 1_000), ("late", 5_000)]


def test_virtual_clock_runs_a_long_wait_without_spending_it():
    async def scenario():
        scheduler = VirtualScheduler()
        scheduler.start()
        await scheduler.sleep_ms(600_000)
        now = scheduler.now_ms()
        await scheduler.stop()
        return now

    assert asyncio.run(scenario()) == 600_000


def test_race_timeout_reports_which_side_won():
    async def scenario():
        scheduler = VirtualScheduler()
        scheduler.start()
        event = asyncio.Event()

        async def set_later():
            await scheduler.sleep_ms(100)
            event.set()

        task = asyncio.create_task(set_later())
        won = await race_timeout(scheduler, event.wait(), 5_000)
        lost = await race_timeout(scheduler, asyncio.Event().wait(), 50)
        await task
        await scheduler.stop()
        return won, lost

    assert asyncio.run(scenario()) == (True, False)


# ---------------------------------------------------------------------------
# turn selection
# ---------------------------------------------------------------------------


def _script(*intents) -> CandidateScript:
    return CandidateScript(
        persona="cooperative",
        turns=[CandidateTurn(f"line {i}", intent) for i, intent in enumerate(intents)],
    )


def test_selector_walks_the_script_in_order_then_stops():
    import random

    selector = TurnSelector(_script("answering", "short"), random.Random(0))
    assert selector.next_turn().text == "line 0"
    assert selector.next_turn().text == "line 1"
    assert selector.next_turn() is None


def test_selector_changes_intent_when_the_agent_repeats_itself():
    """A re-asked question must draw a different kind of answer, or the agent's
    re-asking is invisible in the transcript."""
    import random

    selector = TurnSelector(
        _script("deflecting", "deflecting", "answering"), random.Random(0)
    )
    assert selector.next_turn().intent == "deflecting"
    assert selector.note_agent_line("Tell me about the project.") is False
    assert selector.note_agent_line("Tell me about the project?") is True
    assert selector.next_turn(agent_repeated=True).intent == "answering"


def test_selector_falls_back_to_order_when_no_other_intent_is_left():
    import random

    selector = TurnSelector(_script("vague", "vague"), random.Random(0))
    selector.next_turn()
    assert selector.next_turn(agent_repeated=True).intent == "vague"


# ---------------------------------------------------------------------------
# clip source
# ---------------------------------------------------------------------------


def test_clip_source_caches_so_a_rerun_costs_nothing(tmp_path: Path):
    source = ClipSource(cache_dir=tmp_path, sample_rate=DEFAULT_SAMPLE_RATE)
    persona = get_persona("cooperative")
    turn = get_script("cooperative").turns[0]

    first = source.clip_for(persona, turn, 1)
    second = source.clip_for(persona, turn, 1)
    assert first.data == second.data
    assert source.synthetic == 1
    assert source.cache_hits == 1


# ---------------------------------------------------------------------------
# loopback transport
# ---------------------------------------------------------------------------


def test_loopback_agent_rejects_incoherent_settings():
    with pytest.raises(ValueError):
        LoopbackAgent(join_delay_ms=-1)
    with pytest.raises(ValueError):
        LoopbackAgent(utterance_ms=[])


def test_loopback_emits_silence_frames_while_the_agent_is_quiet():
    """A real subscribed track delivers silence, not nothing. So must this."""

    async def scenario():
        scheduler = VirtualScheduler()
        scheduler.start()
        transport = LoopbackTransport(
            scheduler, LoopbackAgent(join_delay_ms=100, greets=False)
        )
        await transport.connect()
        assert await transport.wait_for_agent(5_000)
        await scheduler.sleep_ms(200)
        await transport.close()
        await scheduler.stop()

        frames = []
        while not transport.agent_audio.empty():
            frames.append(transport.agent_audio.get_nowait())
        return [f for f in frames if f is not None]

    frames = asyncio.run(scenario())
    assert frames, "a live track must deliver frames even in silence"
    assert all(rms(f.pcm) == 0.0 for f in frames)


def test_loopback_never_joining_is_reported_not_hung():
    async def scenario():
        scheduler = VirtualScheduler()
        scheduler.start()
        transport = LoopbackTransport(scheduler, LoopbackAgent(join_delay_ms=60_000))
        await transport.connect()
        joined = await transport.wait_for_agent(1_000)
        await transport.close()
        await scheduler.stop()
        return joined

    assert asyncio.run(scenario()) is False


# ---------------------------------------------------------------------------
# the loop, end to end
# ---------------------------------------------------------------------------


# Sessions here run at 8 kHz rather than the 48 kHz default. Nothing in the loop
# depends on the sample rate -- it changes buffer sizes and nothing else -- while
# the per-sample work in generating and measuring audio scales directly with it,
# so this is six times less arithmetic for identical control flow. 8 kHz is also
# a real telephony rate rather than an invented one. One test below pins the
# 48 kHz default explicitly so the shipped configuration is still covered.
TEST_SAMPLE_RATE = 8_000


def _run(persona: str, tmp_path: Path, agent: LoopbackAgent = None, **overrides):
    overrides.setdefault("sample_rate", TEST_SAMPLE_RATE)
    config = SessionConfig(persona=persona, **overrides)
    return asyncio.run(
        run_session(
            config,
            transport_kind="loopback",
            clock="virtual",
            transcript_dir=tmp_path / "t",
            audio_dir=tmp_path / "a",
            loopback_agent=agent,
            verbose=False,
        )
    )


def test_a_cooperative_session_completes_its_script(tmp_path: Path):
    log = _run("cooperative", tmp_path, max_turns=20)
    assert log.stop_reason == STOP_SCRIPT_EXHAUSTED
    counts = log.to_dict()["counts"]
    assert counts["candidate_turns"] == len(get_script("cooperative").turns)
    assert counts["agent_turns"] > 0
    assert counts["barge_ins"] == 0


def test_the_turn_ceiling_stops_a_session_that_would_run_on(tmp_path: Path):
    log = _run("cooperative", tmp_path, max_turns=2)
    assert log.stop_reason == STOP_TURN_CEILING
    assert log.to_dict()["counts"]["candidate_turns"] == 2


def test_the_wall_clock_stops_a_session_regardless_of_the_conversation(tmp_path: Path):
    log = _run("nervous", tmp_path, max_turns=99, max_wall_clock_ms=4_000)
    assert log.stop_reason == STOP_WALL_CLOCK
    assert log.duration_ms <= 4_000 + 2_000


def test_an_agent_that_never_joins_is_reported(tmp_path: Path):
    log = _run(
        "cooperative",
        tmp_path,
        agent=LoopbackAgent(join_delay_ms=90_000),
        agent_join_timeout_ms=1_000,
    )
    assert log.stop_reason == STOP_AGENT_NEVER_JOINED
    assert log.to_dict()["counts"]["candidate_turns"] == 0


def test_an_agent_that_goes_quiet_is_reported_not_waited_on_forever(tmp_path: Path):
    log = _run(
        "cooperative",
        tmp_path,
        agent=LoopbackAgent(goes_silent_after_turn=1),
        agent_silence_timeout_ms=3_000,
        max_dead_air_strikes=2,
        max_wall_clock_ms=120_000,
    )
    assert log.stop_reason == STOP_AGENT_SILENT
    assert len(log.events_of("dead_air")) == 2


def test_an_aggressive_persona_actually_overlaps_the_agent(tmp_path: Path):
    log = _run("edge_case", tmp_path, max_turns=8)
    counts = log.to_dict()["counts"]
    assert counts["intended_barge_ins"] > 0
    assert counts["barge_ins"] > 0
    assert counts["overlap_ms"] > 0
    # Every confirmed barge-in must have been an intended one: the loop must
    # never overlap the agent by accident.
    for turn in log.turns:
        if turn.actor == ACTOR_CANDIDATE and turn.barge_in:
            assert turn.intended_barge_in


def test_a_calm_persona_never_overlaps_the_agent(tmp_path: Path):
    log = _run("cooperative", tmp_path, max_turns=6)
    for turn in log.turns:
        if turn.actor == ACTOR_CANDIDATE:
            assert turn.overlap_ms == 0
            assert not turn.intended_barge_in


def test_one_interruption_per_agent_utterance(tmp_path: Path):
    """Consecutive candidate turns with no agent speech between them mean the
    loop is machine-gunning into a single utterance."""
    log = _run("edge_case", tmp_path, max_turns=8)
    actors = [t.actor for t in sorted(log.turns, key=lambda t: t.start_ms)]
    runs = max(
        (
            sum(1 for _ in group)
            for actor, group in __import__("itertools").groupby(actors)
            if actor == ACTOR_CANDIDATE
        ),
        default=0,
    )
    assert runs <= 1


def test_the_same_seed_reproduces_the_same_conversation(tmp_path: Path):
    first = _run("difficult", tmp_path / "1", max_turns=5, seed=7)
    second = _run("difficult", tmp_path / "2", max_turns=5, seed=7)
    timeline = lambda log: [(e.t_ms, e.kind) for e in log.events if e.kind != "session_start"]
    assert timeline(first) == timeline(second)


def test_a_different_seed_changes_an_interrupting_persona(tmp_path: Path):
    a = _run("difficult", tmp_path / "a", max_turns=6, seed=1)
    b = _run("difficult", tmp_path / "b", max_turns=6, seed=4)
    intents_a = [t.intended_barge_in for t in a.turns if t.actor == ACTOR_CANDIDATE]
    intents_b = [t.intended_barge_in for t in b.turns if t.actor == ACTOR_CANDIDATE]
    assert intents_a != intents_b


def test_a_virtual_session_is_marked_as_not_measurable(tmp_path: Path):
    """Nothing downstream may mistake simulated time for a measurement."""
    log = _run("cooperative", tmp_path, max_turns=2)
    assert log.to_dict()["time_scale"] == 0.0


def test_the_session_writes_the_artifacts_week_three_will_read(tmp_path: Path):
    log = _run("cooperative", tmp_path, max_turns=3)
    record = tmp_path / "t" / f"{log.session_id}_cooperative.json"
    assert record.exists()

    data = load_session(record)
    assert data["transport"] == "loopback"
    assert data["events"] and data["turns"]
    assert Path(data["config"]["agent_audio_wav"]).exists()
    assert Path(data["config"]["candidate_audio_wav"]).exists()
    # Both sides are recorded, and every event carries a timestamp.
    assert all("t_ms" in e for e in data["events"])
    assert {t["actor"] for t in data["turns"]} == {ACTOR_AGENT, ACTOR_CANDIDATE}


def test_the_shipped_48khz_configuration_runs(tmp_path: Path):
    """The other session tests run at 8 kHz for speed; the default must work."""
    log = _run("cooperative", tmp_path, max_turns=3, sample_rate=DEFAULT_SAMPLE_RATE)
    assert log.stop_reason == STOP_TURN_CEILING
    record = load_session(tmp_path / "t" / f"{log.session_id}_cooperative.json")
    assert record["config"]["sample_rate"] == DEFAULT_SAMPLE_RATE
    assert PcmClip.from_wav(
        Path(record["config"]["candidate_audio_wav"])
    ).sample_rate == DEFAULT_SAMPLE_RATE


def test_every_persona_survives_a_session(tmp_path: Path):
    """A persona that cannot complete a session is a broken persona."""
    from persona_spec import PERSONAS

    for name in sorted(PERSONAS):
        log = _run(name, tmp_path / name, max_turns=4, max_wall_clock_ms=120_000)
        assert log.stop_reason in (STOP_SCRIPT_EXHAUSTED, STOP_TURN_CEILING), name
        assert log.error is None, name


# ---------------------------------------------------------------------------
# agent dispatch
# ---------------------------------------------------------------------------


def _decode(token: str) -> dict:
    import jwt

    return jwt.decode(token, options={"verify_signature": False})


def test_token_carries_no_dispatch_when_no_agent_is_named():
    from livekit import api

    from transport import LiveKitConfig, build_access_token

    claims = _decode(
        build_access_token(
            api,
            LiveKitConfig(
                url="ws://x", api_key="k", api_secret="s" * 32, room="harness"
            ),
        )
    )
    assert claims["video"]["room"] == "harness"
    assert not claims.get("roomConfig")


def test_token_requests_the_named_agent_with_its_metadata():
    """A worker started with an agent_name is never auto-dispatched. If this
    request is missing, the session reports agent_never_joined and says nothing
    about why."""
    from livekit import api

    from transport import LiveKitConfig, build_access_token

    claims = _decode(
        build_access_token(
            api,
            LiveKitConfig(
                url="ws://x",
                api_key="k",
                api_secret="s" * 32,
                room="harness",
                agent_name="interview-bot",
                agent_metadata='{"meeting_id": "abc"}',
            ),
        )
    )
    agents = claims["roomConfig"]["agents"]
    assert len(agents) == 1
    assert agents[0]["agentName"] == "interview-bot"
    assert json.loads(agents[0]["metadata"])["meeting_id"] == "abc"


def test_session_config_records_the_agent_but_never_its_metadata(tmp_path: Path):
    """Metadata routinely holds an operator's prompt; records get pasted around."""
    secret = '{"system_prompt": "SHOULD-NOT-APPEAR-IN-ANY-RECORD"}'
    config = SessionConfig(
        persona="cooperative", agent_name="interview-bot", agent_metadata=secret
    )
    rendered = json.dumps(config.to_dict())
    assert "interview-bot" in rendered
    assert "SHOULD-NOT-APPEAR-IN-ANY-RECORD" not in rendered
    assert config.to_dict()["agent_metadata_bytes"] == len(secret)


def test_agent_metadata_file_must_be_valid_json(tmp_path: Path):
    from audio_loop import load_agent_metadata

    bad = tmp_path / "meta.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_agent_metadata(bad)

    missing = tmp_path / "nope.json"
    with pytest.raises(ValueError, match="not found"):
        load_agent_metadata(missing)

    good = tmp_path / "ok.json"
    good.write_text('{"meeting_id": "m1"}', encoding="utf-8")
    assert json.loads(load_agent_metadata(good))["meeting_id"] == "m1"


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def test_env_file_is_actually_read(tmp_path: Path, monkeypatch):
    """A documented, gitignored .env that nothing loads produces a credentials
    error pointing at the environment while the values sit correct on disk."""
    from private_config import load_env_file

    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "\n"
        "LIVEKIT_URL=wss://example.livekit.cloud\n"
        'LIVEKIT_API_KEY="quoted-key"\n'
        "export LIVEKIT_API_SECRET=exported-secret\n"
        "MALFORMED_LINE\n",
        encoding="utf-8",
    )
    for k in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(k, raising=False)

    assert load_env_file(env) == 3
    import os

    assert os.environ["LIVEKIT_URL"] == "wss://example.livekit.cloud"
    assert os.environ["LIVEKIT_API_KEY"] == "quoted-key"     # quotes stripped
    assert os.environ["LIVEKIT_API_SECRET"] == "exported-secret"  # export handled


def test_env_file_does_not_clobber_a_real_export(tmp_path: Path, monkeypatch):
    """An explicit export or a CI secret must beat a stale file on disk."""
    from private_config import load_env_file

    env = tmp_path / ".env"
    env.write_text("LIVEKIT_URL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("LIVEKIT_URL", "from-environment")

    assert load_env_file(env) == 0
    import os

    assert os.environ["LIVEKIT_URL"] == "from-environment"
    assert load_env_file(env, override=True) == 1
    assert os.environ["LIVEKIT_URL"] == "from-file"


def test_missing_env_file_is_not_an_error(tmp_path: Path):
    from private_config import load_env_file

    assert load_env_file(tmp_path / "nope.env") == 0
