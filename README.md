# duplex-harness

A regression harness for real-time voice agents.

Voice agents are tested by hand. Someone changes an instruction, places a few
calls, listens, and forms an opinion. That does not scale, it is not
reproducible, and it misses the failures that only exist in the audio path:
transcription errors, endpointing that fires too early, barge-in handled
inconsistently, latency that varies by language.

This harness drives a simulated speaker through **real audio** into a live
agent, captures both sides with timestamps, and turns the result into numbers
that can be compared across runs and gated in CI.

Text-based simulation cannot do this. A text simulator exercises the model. An
audio harness exercises the system.

## The four layers

| Layer | What it does | Status |
|-------|--------------|--------|
| 1. Personas and rendering | Define how a speaker sounds; render their turns to audio | **Working** |
| 2. Audio loop | Publish that audio to a live agent and capture the reply | Planned |
| 3. Metrics | Persist per-turn timing and aggregate it per language | Planned |
| 4. Behaviour gates | Check transcripts against behavioural rules, fail the build | Planned |

## Design

A **persona** is *how* someone speaks: language, accent, speech rate, how long
they pause before replying, how readily they interrupt, whether they switch
language mid-sentence. A **script** is *what* they say, each turn tagged by
intent. The two are separate, so any script can be delivered by any persona —
the same words at a different pace exercise different code paths in the agent.

Seven personas ship, each targeting a specific class of failure:

| Persona | Probes |
|---------|--------|
| `cooperative` | Control case |
| `nervous` | Long mid-answer pauses — does the agent wait, or cut in? |
| `difficult` | Deflection — does the agent re-ask, or silently move on? |
| `edge_case` | Aggressive interruption and 9-second silences |
| `adversarial` | Guardrails, attacked through transcribed speech rather than text |
| `hindi_multilingual` | Mid-sentence code switching |
| `spanish_accent` | Correct language, non-dominant phonetics |

## Quick start

Everything in layer 1 runs offline. No account, no API key, no network.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# Inspect what is defined
python harness/persona_spec.py         # persona table
python harness/candidate_scripts.py    # all scripts

# Exercise the full pipeline with no API calls
python harness/run_benchmark.py --mode tts --dry-run

# Tests (58, all offline)
python -m pytest
```

Rendering real audio needs a TTS key. Nothing else about the project requires
one, and the default is always dry run:

```bash
export CARTESIA_API_KEY=...
export CARTESIA_VOICE_ID=...
python harness/persona_tts.py --persona cooperative --output /tmp/check.mp3
```

## Layout

```
harness/
  persona_spec.py        how a simulated speaker sounds        (layer 1)
  candidate_scripts.py   what they say, tagged by intent       (layer 1)
  persona_tts.py         render turns to audio                 (layer 1)
  audio_loop.py          publish to an agent, capture reply    (layer 2)
  metrics_extractor.py   timing analysis                       (layer 3)
  prompt_judge.py        behavioural gates                     (layer 4)
  run_benchmark.py       entry point
tests/
  test_week1.py          58 offline tests
results/                 generated artifacts (audio is not committed)
```

`harness/` is a plain directory on the path rather than an installed package, so
each module runs directly as a script and `pytest.ini` puts it on `pythonpath`
for tests. No `sys.path` manipulation anywhere.

## Dependencies

One: `pytest`. The TTS backend speaks HTTP over the standard library rather than
importing a vendor SDK — no wheel-availability risk, no transitive dependency
surface, and the single place that encodes provider-specific field names is one
testable function.

Backends sit behind a small protocol, so a local engine can replace the hosted
one without touching anything above it.

## Scope

A personal project. Not affiliated with any employer, no dependency on any
employer system, and no proprietary prompt, rubric, question bank, or candidate
data. It is built to run against **any** LiveKit or Pipecat agent.

The behavioural rules in layer 4 are written for this project, against
behaviours generic to interview-style voice agents. A private rule set can be
supplied at runtime — that is where anything proprietary belongs, never in this
repository.

See [INFRA.md](INFRA.md) for the infrastructure rules enforced for every service
and dependency this project touches.

## Licence

MIT.
