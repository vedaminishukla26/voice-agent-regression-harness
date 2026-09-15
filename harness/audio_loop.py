"""Week 2: close the audio loop.

Publishes rendered persona audio into a real room, lets the agent under test
respond over its own pipeline, and captures both sides with timestamps.

This is the layer that makes the harness worth building. Everything in week 1
could be done against text; nothing here can. Publishing real audio is what
exercises speech-to-text, voice-activity detection, endpointing, barge-in and
text-to-speech -- the components where the interesting regressions actually live.

Design notes carried forward, so the shape is decided before the code lands:

* Target a locally run media server by default. A local server needs no account
  and no hosted project, which keeps this layer free of external infrastructure.
* The known hazard is two agents deadlocking or feeding back into each other.
  Every session needs a hard wall-clock timeout and a turn ceiling, independent
  of whatever the conversation thinks it is doing.
* Capture must be timestamped at the point of arrival, not reconstructed after
  the fact. Reconstructed timing cannot measure barge-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def run_audio_session(persona_name: str, room_name: str, out_dir: Path) -> Dict:
    """Run one full-duplex session against a live agent. Not yet implemented."""
    raise NotImplementedError(
        "Week 2. Requires a running media server and an agent under test."
    )
