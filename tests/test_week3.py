"""Week 3 tests: deriving comparable numbers from captured sessions.

Sessions are built inline as dicts rather than by running conversations. The
metrics layer's contract is with the *record format*, not with the loop, and
stating the timeline explicitly is what makes each expected latency obvious by
inspection instead of something the test has to trust.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metrics_extractor import (
    LATENCY_COMPONENTS,
    MetricsError,
    aggregate,
    compare,
    extract_session_metrics,
    load_sessions,
    percentile,
    summarise,
    write_outputs,
)
from session_log import (
    ACTOR_AGENT,
    ACTOR_CANDIDATE,
    AGENT_SPEECH_END,
    AGENT_SPEECH_START,
    BARGE_IN,
    CANDIDATE_SPEECH_END,
    CANDIDATE_SPEECH_START,
    STOP_SCRIPT_EXHAUSTED,
    STOP_WALL_CLOCK,
)


def _event(t_ms, kind, actor, **data):
    return {"t_ms": t_ms, "kind": kind, "actor": actor, "data": data}


def _session(events, turns=(), persona="cooperative", time_scale=1.0, **over):
    base = {
        "session_id": "test01",
        "persona": persona,
        "transport": "livekit",
        "time_scale": time_scale,
        "stop_reason": STOP_SCRIPT_EXHAUSTED,
        "duration_ms": max([e["t_ms"] for e in events], default=0),
        "events": events,
        "turns": list(turns),
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------


def test_percentile_returns_an_observed_value_not_an_invented_one():
    """Nearest-rank: with few samples, interpolation reports a number nobody saw."""
    values = [100, 200, 300, 400]
    assert percentile(values, 50) in values
    assert percentile(values, 95) == 400
    assert percentile(values, 0) == 100
    assert percentile([], 50) is None


def test_summarise_reports_n_so_a_reader_can_weigh_it():
    s = summarise([10, 20, 30])
    assert s["n"] == 3 and s["median_ms"] == 20
    empty = summarise([])
    assert empty["n"] == 0 and empty["median_ms"] is None


# ---------------------------------------------------------------------------
# the simulation guard
# ---------------------------------------------------------------------------


def test_simulated_sessions_are_refused_by_default():
    """A virtual-clock session is exact simulation, not measurement."""
    session = _session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)], time_scale=0.0)
    with pytest.raises(MetricsError, match="simulated time"):
        extract_session_metrics(session)


def test_simulated_sessions_can_be_opted_into_explicitly():
    session = _session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)], time_scale=0.0)
    assert extract_session_metrics(session, allow_simulated=True).persona == "cooperative"


def test_real_sessions_need_no_opt_in():
    session = _session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)])
    assert extract_session_metrics(session).transport == "livekit"


# ---------------------------------------------------------------------------
# response latency
# ---------------------------------------------------------------------------


def test_response_latency_is_candidate_end_to_agent_start():
    session = _session(
        [
            _event(0, CANDIDATE_SPEECH_START, ACTOR_CANDIDATE),
            _event(1_000, CANDIDATE_SPEECH_END, ACTOR_CANDIDATE),
            _event(1_800, AGENT_SPEECH_START, ACTOR_AGENT),   # 800 ms
            _event(4_000, AGENT_SPEECH_END, ACTOR_AGENT),
            _event(5_000, CANDIDATE_SPEECH_START, ACTOR_CANDIDATE),
            _event(6_000, CANDIDATE_SPEECH_END, ACTOR_CANDIDATE),
            _event(7_200, AGENT_SPEECH_START, ACTOR_AGENT),   # 1200 ms
        ]
    )
    assert extract_session_metrics(session).response_latency_ms == [800, 1_200]


def test_latency_is_not_invented_across_a_turn_the_agent_never_answered():
    """If the candidate speaks twice with no reply between, there is no latency
    to record -- pairing across it would span unrelated events."""
    session = _session(
        [
            _event(0, CANDIDATE_SPEECH_START, ACTOR_CANDIDATE),
            _event(1_000, CANDIDATE_SPEECH_END, ACTOR_CANDIDATE),
            # agent stays silent; candidate speaks again
            _event(9_000, CANDIDATE_SPEECH_START, ACTOR_CANDIDATE),
            _event(10_000, CANDIDATE_SPEECH_END, ACTOR_CANDIDATE),
            _event(10_600, AGENT_SPEECH_START, ACTOR_AGENT),  # 600 ms, the only one
        ]
    )
    assert extract_session_metrics(session).response_latency_ms == [600]


def test_an_agent_that_never_replies_yields_no_latency_samples():
    session = _session(
        [
            _event(0, CANDIDATE_SPEECH_START, ACTOR_CANDIDATE),
            _event(1_000, CANDIDATE_SPEECH_END, ACTOR_CANDIDATE),
        ]
    )
    metrics = extract_session_metrics(session)
    assert metrics.response_latency_ms == []
    assert summarise(metrics.response_latency_ms)["median_ms"] is None


# ---------------------------------------------------------------------------
# barge-in yield
# ---------------------------------------------------------------------------


def test_barge_in_yield_measures_how_fast_the_agent_stops():
    session = _session(
        [
            _event(0, AGENT_SPEECH_START, ACTOR_AGENT),
            _event(700, BARGE_IN, ACTOR_CANDIDATE),
            _event(700, CANDIDATE_SPEECH_START, ACTOR_CANDIDATE),
            _event(1_000, AGENT_SPEECH_END, ACTOR_AGENT),   # yielded after 300 ms
        ]
    )
    assert extract_session_metrics(session).barge_in_yield_ms == [300]


def test_a_barge_in_with_no_open_agent_turn_is_not_a_yield():
    """An interruption that missed produced no overlap, so there is nothing to
    time -- counting it would report a yield the agent never performed."""
    session = _session(
        [
            _event(0, AGENT_SPEECH_START, ACTOR_AGENT),
            _event(500, AGENT_SPEECH_END, ACTOR_AGENT),
            _event(900, BARGE_IN, ACTOR_CANDIDATE),   # agent already silent
            _event(2_000, AGENT_SPEECH_START, ACTOR_AGENT),
            _event(3_000, AGENT_SPEECH_END, ACTOR_AGENT),
        ]
    )
    assert extract_session_metrics(session).barge_in_yield_ms == []


# ---------------------------------------------------------------------------
# turn-derived figures
# ---------------------------------------------------------------------------


def test_turn_counts_overlap_and_barge_in_rate():
    turns = [
        {"actor": ACTOR_AGENT, "duration_ms": 2_000, "overlap_ms": None},
        {
            "actor": ACTOR_CANDIDATE,
            "duration_ms": 900,
            "overlap_ms": 250,
            "intended_barge_in": True,
            "barge_in": True,
        },
        {
            "actor": ACTOR_CANDIDATE,
            "duration_ms": 700,
            "overlap_ms": 0,
            "intended_barge_in": True,
            "barge_in": False,   # aimed and missed
        },
    ]
    m = extract_session_metrics(_session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)],
                                         turns=turns, duration_ms=10_000))
    assert m.agent_turns == 1 and m.candidate_turns == 2
    assert m.overlap_ms == 250
    assert m.intended_barge_ins == 2 and m.confirmed_barge_ins == 1
    assert m.barge_in_success_rate == 0.5
    assert m.overlap_ratio == 0.025


def test_unhealthy_stop_reason_is_flagged():
    session = _session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)],
                       stop_reason=STOP_WALL_CLOCK)
    assert extract_session_metrics(session).healthy is False


def test_language_comes_from_the_persona_definition():
    hindi = _session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)],
                     persona="hindi_multilingual")
    assert extract_session_metrics(hindi).language == "hi"
    unknown = _session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)], persona="nope")
    assert extract_session_metrics(unknown).language == "unknown"


# ---------------------------------------------------------------------------
# aggregation and comparison
# ---------------------------------------------------------------------------


def _latency_session(persona, latencies, lag=40):
    """Build a session whose response latencies are exactly ``latencies``."""
    events, t = [], 0
    for value in latencies:
        events.append(_event(t, CANDIDATE_SPEECH_START, ACTOR_CANDIDATE))
        events.append(_event(t + 500, CANDIDATE_SPEECH_END, ACTOR_CANDIDATE))
        events.append(
            _event(t + 500 + value, AGENT_SPEECH_START, ACTOR_AGENT,
                   detection_lag_ms=lag)
        )
        events.append(_event(t + 500 + value + 1_500, AGENT_SPEECH_END, ACTOR_AGENT,
                             detection_lag_ms=lag))
        t += 500 + value + 2_000
    return _session(events, persona=persona, duration_ms=t)


def test_aggregate_groups_by_language():
    records = [
        extract_session_metrics(_latency_session("cooperative", [800, 900])),
        extract_session_metrics(_latency_session("hindi_multilingual", [1_400, 1_500])),
    ]
    grouped = aggregate(records, by="language")
    assert set(grouped) == {"en", "hi"}
    assert grouped["en"]["response_latency"]["n"] == 2
    assert grouped["hi"]["response_latency"]["median_ms"] == 1_400
    assert grouped["en"]["latency_contains"] == LATENCY_COMPONENTS


def test_aggregate_rejects_an_unknown_grouping():
    with pytest.raises(MetricsError, match="cannot group by"):
        aggregate([], by="colour")


def test_compare_reports_a_difference_against_a_baseline():
    records = [
        extract_session_metrics(_latency_session("cooperative", [800, 800])),
        extract_session_metrics(_latency_session("hindi_multilingual", [1_300, 1_300])),
    ]
    result = compare(aggregate(records), baseline="en")
    assert result["hi"]["delta_vs_baseline_ms"] == 500
    assert result["hi"]["above_detector_floor"] is True


def test_a_difference_inside_detector_resolution_is_not_called_evidence():
    """The whole project's headline is a per-language latency difference, so a
    gap smaller than the detector's own lag must not be reported as a finding."""
    records = [
        extract_session_metrics(_latency_session("cooperative", [800], lag=400)),
        extract_session_metrics(_latency_session("hindi_multilingual", [850], lag=400)),
    ]
    result = compare(aggregate(records), baseline="en")
    assert result["hi"]["delta_vs_baseline_ms"] == 50
    assert result["hi"]["above_detector_floor"] is False


def test_compare_rejects_a_missing_baseline():
    records = [extract_session_metrics(_latency_session("cooperative", [800]))]
    with pytest.raises(MetricsError, match="not present"):
        compare(aggregate(records), baseline="hi")


# ---------------------------------------------------------------------------
# directory loading and output
# ---------------------------------------------------------------------------


def test_load_sessions_skips_unreadable_and_simulated_with_reasons(tmp_path: Path):
    good = _latency_session("cooperative", [800])
    (tmp_path / "good.json").write_text(json.dumps(good), encoding="utf-8")
    sim = _session([_event(0, AGENT_SPEECH_START, ACTOR_AGENT)], time_scale=0.0)
    (tmp_path / "sim.json").write_text(json.dumps(sim), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")

    records, skipped = load_sessions(tmp_path)
    assert len(records) == 1
    assert len(skipped) == 2
    assert any("simulated time" in s for s in skipped)
    assert any("unreadable" in s for s in skipped)


def test_outputs_are_written_and_reloadable(tmp_path: Path):
    records = [extract_session_metrics(_latency_session("cooperative", [800, 900]))]
    json_path, csv_path = write_outputs(records, aggregate(records), tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["sessions"][0]["response_latency"]["n"] == 2
    assert payload["latency_contains"] == LATENCY_COMPONENTS

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "latency_median_ms" in lines[0]
