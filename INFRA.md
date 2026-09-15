# Infrastructure policy

This project deliberately runs on as little infrastructure as possible. Every
external service is a cost, a dependency, an account to maintain, and a reason
the test suite eventually stops being run. The default answer to "should this
call out to something?" is no.

## Principles

**Local before hosted.** If a capability can run as a local process, it does.
The media server for the audio loop is intended to run locally in dev mode, with
throwaway keys, so no hosted project or account is required to use this harness.

**Offline by default.** Layer 1 runs with no API key, no network, and no
account. `--dry-run` is the default path, not an afterthought, so the pipeline
can always be exercised for free. A test suite that requires billing is a test
suite that rots.

**Filesystem, not object storage.** Session artifacts are written to
`results/`. There is no bucket, no upload, and no retention policy to manage.

**Files, not a database.** State is JSON on disk. Nothing here connects to a
database, and nothing here should start.

**No vendor SDKs where HTTP will do.** The TTS backend speaks HTTP over the
standard library. Total runtime dependency count for layer 1 is zero; the only
requirement is a test runner.

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
