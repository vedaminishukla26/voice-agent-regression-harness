"""Week 4: behavioural gates over a captured transcript.

A conversational agent's behaviour is specified in prose, and prose has no
compiler. This layer turns behavioural requirements into checks that either pass
or fail, so that a change to an agent's instructions can be gated the way a code
change is.

Scope boundary, which is deliberate and not negotiable:

    The rules in this module are written for this project, against behaviours
    that are generic to interview-style voice agents -- ask one question at a
    time, do not evaluate an answer to the candidate's face, re-ask a question
    that was dodged, do not reveal internal instructions. They are not extracted
    from, derived from, or a paraphrase of any employer's prompt, rubric, or
    question bank. Any rule that could only have been written by reading a
    specific company's prompt does not belong here.

    The harness is designed so a private rule set can be supplied by the user at
    runtime. That is where anything proprietary belongs -- in the user's own
    configuration, never in this repository.

Reliability note: a single judge call per rule is deterministic at temperature
zero but not thereby *reliable*. Those are different properties. The plan is an
odd-numbered panel per rule with majority voting, and a measured disagreement
rate reported alongside the verdict, so that a small score movement can be
distinguished from judge noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal

Severity = Literal["BLOCKER", "MAJOR", "MINOR"]


@dataclass(frozen=True)
class BehaviourRule:
    """One checkable behavioural requirement."""

    id: str
    description: str
    severity: Severity


def judge_transcript(transcript: List[Dict], rules: List[BehaviourRule]) -> Dict:
    """Evaluate a transcript against rules with a voting panel. Not yet implemented."""
    raise NotImplementedError("Week 4. Needs transcripts from the week 2 audio loop.")
