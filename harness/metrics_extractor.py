"""Week 3: extract timing from captured sessions.

Agent frameworks commonly emit rich per-turn metrics -- end-of-utterance delay,
time to first token, time to first audio byte -- and then log them to stdout,
where they are read once by a human and lost. This layer persists them, joins
them to the persona that produced them, and aggregates across sessions so a
change in behaviour is visible as a change in a number.

Metrics to derive:

* ``barge_in_latency_ms``  -- speaker starts talking over the agent, to agent silence.
* ``eou_to_ttft_ms``       -- end of user utterance, to first model token.
* ``ttft_to_ttfb_ms``      -- first model token, to first audio byte out.
* ``overlap_ratio``        -- share of session where both parties spoke at once.
* ``dead_air_ms``          -- distribution of gaps where neither party spoke.

The comparison that matters is per language. A single global endpointing
threshold is the norm in agent implementations, and a threshold tuned against
one language is not automatically right for the others it serves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def extract_session_metrics(session_path: Path) -> Dict:
    """Parse one captured session into a flat metrics record. Not yet implemented."""
    raise NotImplementedError("Week 3. Needs sessions from the week 2 audio loop.")


def aggregate(records: List[Dict]) -> Dict:
    """Aggregate per-session records by persona and language. Not yet implemented."""
    raise NotImplementedError("Week 3.")
