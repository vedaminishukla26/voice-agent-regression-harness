# Infrastructure policy

This project deliberately runs on as little infrastructure as possible. Every
external service is a cost, a dependency, an account to maintain, and a reason
the test suite eventually stops being run. The default answer to "should this
call out to something?" is no.

## Principles

**Local before hosted.** If a capability can run as a local process, it does.
The media server for the audio loop runs locally in dev mode with throwaway
keys, so no hosted project or account is required to use this harness.

**Offline by default.** Layers 1 and 2 both run with no API key, no network, and
no account. For layer 1 that is `--dry-run`; for layer 2 it is the loopback
transport, which simulates an agent in-process. Neither is an afterthought:
they are the default path, so the pipeline can always be exercised for free.
A test suite that requires billing is a test suite that rots.

**And a test suite that is slow rots the same way.** Sessions are minutes of
mostly waiting, so the suite runs them under a simulated clock that advances to
the next scheduled event instead of sleeping. The obvious alternative — dividing
real sleeps by a speed factor — does not work: timer resolution is around 15 ms
on Windows, so a 1 ms sleep takes fifteen, and the audio frames the detector
sees end up spaced far wider than the thresholds it counts in frames. That does
not merely distort the numbers, it changes the behaviour, leaving the tests
exercising a state machine that is not the one that ships. Simulated time is
exact; compressed real time is neither exact nor honest about it.

**Filesystem, not object storage.** Session artifacts are written to
`results/`. There is no bucket, no upload, and no retention policy to manage.

**Files, not a database.** State is JSON on disk. Nothing here connects to a
database, and nothing here should start.

**No vendor SDKs where HTTP will do.** The TTS backend speaks HTTP over the
standard library. Total runtime dependency count for layer 1 is zero; the only
requirement is a test runner, and the whole test suite still runs on that alone.

Layer 2 adds exactly one, `livekit`, because speaking WebRTC by hand is not
reasonable. It is imported lazily inside the transport that needs it, so the
offline path neither loads nor requires it. Two further dependencies that would
normally be assumed are absent by construction: no MP3 decoder, because the
provider is asked for raw PCM rather than a compressed container; and no
resampler, because audio is requested at the transport's own rate. `audioop`
would have been the standard-library answer to the second and was removed in
Python 3.13 — a useful reminder that the cheapest dependency is the one the
design makes unnecessary.

**The agent under test is never part of this project.** The harness joins a room
and behaves like a candidate; the agent is a separate process that joins the
same room. Nothing here imports it, launches it, or stores its prompt or model
configuration. That keeps the harness usable against any agent on the platform,
and it means testing a proprietary agent requires nothing proprietary to enter
this repository.

## Provider accounts

Only the model providers genuinely require an account, and only for layers that
touch real audio:

| Capability | Local option | Hosted option |
|-----------|--------------|---------------|
| Media server | dev-mode server, throwaway keys | hosted project |
| Text to speech | local engine | hosted API + key |
| Speech to text | local model | hosted API + key |
| Judge model | — | hosted API + key |

Backends sit behind small protocols precisely so the local option remains
viable. Where a hosted provider is used, credentials come from the environment
and never from source. `.env` is gitignored; only `.env.example` is committed.

## Content boundaries

This is an independent project. It carries no third-party proprietary content:
no prompt text, rubric, scoring criteria, question bank, or conversational data
originating from anyone else. Every persona, script, and behavioural rule here
was written for this repository.

The behavioural gates in layer 4 ship only rules that are generic to
interview-style voice agents — ask one question at a time, do not grade an
answer to the speaker's face, re-ask a question that was dodged, do not disclose
internal instructions. The harness is designed to load a private rule set at
runtime, which is where anything confidential belongs: in the operator's own
configuration, never in this repository.

No transcript, recording, or personal data belonging to a real person should
ever be committed here. The personas are synthetic and that is the point.

## Where private material goes

Outside the repository. `harness/private_config.py` resolves a private directory
— `~/.duplex-harness` by default, overridable with `DUPLEX_HARNESS_PRIVATE_DIR`
— and that is where an operator's own behavioural rules, internal settings and
anything confidential belong.

This is deliberately not "add it to `.gitignore`". An ignore rule is a good
second line of defence and a bad first one. It is defeated by `git add -f`, it
silently stops protecting every file it covered the moment someone rewrites it,
and it is not consulted at all by editors, indexers, backup agents, or any other
tool that reads the working tree. A file that is never in the tree cannot be
committed by any of those routes. The ignore rules in this repository exist to
catch the case where something ends up here despite the above, and
`private_config.py` warns if the configured directory turns out to sit inside
the project, because at that point the ignore file is all that is left.

Credentials follow the same rule and come from the environment. `.env` is
gitignored; only `.env.example` is committed, and it holds no values.
