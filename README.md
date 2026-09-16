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
| 2. Audio loop | Publish that audio to a live agent and capture the reply | **Working** |
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

## The audio loop

The harness joins a room and behaves like a candidate. **The agent under test is
a separate process.** Nothing here imports it, starts it, or holds its prompt or
model configuration — it joins the same room by whatever means it normally
would, and the harness listens. That boundary is what makes this runnable
against any agent on the platform, and what keeps anything proprietary out of
this repository entirely.

Three properties are load-bearing.

**Every limit is a wall clock, not a conversational judgement.** Two systems
that each wait for the other will wait forever; two that each react to the other
can feed back indefinitely. Careful turn-taking does not prevent either, because
both arise precisely when the turn-taking has already become confused. So a
session is bounded by elapsed time, a turn ceiling, and an agent-silence
timeout, each enforced regardless of what the conversation believes.

**Timing is recorded, never reconstructed.** Barge-in exists only as an overlap
between two live events; it cannot be recovered afterwards from durations or a
transcript. Speech boundaries are timestamped at the moment they happen, not the
moment the detector becomes sure of them — the detector is allowed to be slow to
decide, never wrong about when.

**An interruption that missed is not an interruption.** The harness records what
it *intended* (talk over the agent) separately from what actually *overlapped*,
resolved at session end once every utterance is closed. Conflating them produces
a barge-in count that is quietly wrong, which is worse than not having one.

## Quick start

Layers 1 and 2 both run offline. No account, no API key, no network, no server.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# Inspect what is defined
python harness/persona_spec.py         # persona table
python harness/candidate_scripts.py    # all scripts

# Layer 1: render every turn, no API calls
python harness/run_benchmark.py --mode tts --dry-run

# Layer 2: a full conversation against a simulated agent, in simulated time.
# Three minutes of dialogue in a fraction of a second.
python harness/audio_loop.py --persona edge_case --clock virtual

# Every persona through the audio loop
python harness/run_benchmark.py --mode audio --transport loopback --clock virtual

# Tests (111, all offline, no sleeping through conversation time)
python -m pytest
```

The loopback transport is a configurable opponent, not a mock that always
succeeds: join delay, response latency, utterance length, whether it yields when
talked over, and whether it goes silent partway through are all settable. That
is what makes the failure paths — an agent that never joins, one that stops
responding, a session that overruns — testable without a server.

### Against a real agent

Start a media server, start the agent under test pointed at the same room, then
put the harness in it:

```bash
livekit-server --dev                       # throwaway keys, no account
# ... start your agent, joining room "harness" ...

cp .env.example .env                       # fill in LIVEKIT_* if not using dev defaults
python harness/audio_loop.py --persona hindi_multilingual \
    --transport livekit --room harness
```

Real speech instead of shaped tones needs a TTS key, and is the only thing that
does. Rendered audio is cached per turn, so a rerun of the same benchmark costs
nothing and replays byte-identical audio:

```bash
export CARTESIA_API_KEY=... CARTESIA_VOICE_ID=...
python harness/audio_loop.py --persona cooperative --transport livekit --tts
```

Each session writes a timestamped JSON record plus both sides of the
conversation as WAV. That record is what layer 3 reads; it is not re-derived
from the audio.

## Layout

```
harness/
  persona_spec.py        how a simulated speaker sounds        (layer 1)
  candidate_scripts.py   what they say, tagged by intent       (layer 1)
  persona_tts.py         render turns to audio                 (layer 1)
  audio_io.py            PCM frames, WAV, offline audio        (layer 2)
  vad.py                 speech boundaries, timestamped        (layer 2)
  session_log.py         the session record; clocks            (layer 2)
  transport.py           LiveKit room, and an offline agent    (layer 2)
  audio_loop.py          turn taking, barge-in, hard limits    (layer 2)
  private_config.py      where operator-private material lives
  metrics_extractor.py   timing analysis                       (layer 3)
  prompt_judge.py        behavioural gates                     (layer 4)
  run_benchmark.py       entry point
tests/
  test_week1.py          58 offline tests
  test_week2.py          53 offline tests
results/                 generated artifacts (audio is not committed)
examples/                one committed session record, for shape
```

`harness/` is a plain directory on the path rather than an installed package, so
each module runs directly as a script and `pytest.ini` puts it on `pythonpath`
for tests. No `sys.path` manipulation anywhere.

## Dependencies

The entire test suite runs on `pytest` alone. The TTS backend speaks HTTP over
the standard library rather than importing a vendor SDK — no wheel-availability
risk, no transitive dependency surface, and the single place that encodes
provider-specific field names is one testable function.

One vendor SDK is unavoidable: `livekit`, for the real transport. Speaking
WebRTC by hand is not reasonable. It is imported lazily inside that transport,
so the offline path never loads it and does not require it to be installed.

Two dependencies that would normally be taken for granted are absent by
construction rather than by effort. There is no MP3 decoder, because the TTS
provider is asked for raw PCM instead of a compressed container. There is no
resampler, because audio is requested at the transport's own rate — `audioop`
would have been the standard-library answer and it was removed in Python 3.13,
which is a fair illustration of the value of not needing it.

Backends and transports sit behind small protocols, so a local engine or a
different platform can replace the hosted one without touching anything above.

## Scope

A personal project. Not affiliated with any employer, no dependency on any
employer system, and no proprietary prompt, rubric, question bank, or candidate
data. It is built to run against **any** LiveKit or Pipecat agent.

The behavioural rules in layer 4 are written for this project, against
behaviours generic to interview-style voice agents. A private rule set can be
supplied at runtime — that is where anything proprietary belongs, never in this
repository.

That runtime location defaults to `~/.duplex-harness`, **outside the repository
by design** (`harness/private_config.py`). A `.gitignore` entry is a good second
line of defence and a poor first one: it does not survive `git add -f`, a
rewritten ignore file, or any editor, indexer or backup agent that reads the
working tree without consulting it. A file that was never in the tree cannot be
committed by any of those routes. The ignore rules exist to catch the case where
something lands there anyway.

See [INFRA.md](INFRA.md) for the infrastructure rules enforced for every service
and dependency this project touches.

## Licence

MIT.
