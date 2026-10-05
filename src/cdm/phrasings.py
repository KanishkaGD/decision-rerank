"""
The relevance-question phrasings for System One models (Laya, Jev), which are asked a question about each
(query, passage) pair rather than producing a learned relevance score.

Protocol (research plan, "Question-phrasing protocol"): fix 3-5 phrasings in advance, select one on the dev split
only (msmarco-dev or a held-out BEIR set), report all of them. The texts below are a DRAFT until WP3 freezes them;
once frozen, never edit a phrasing in place -- add a new id, so results.csv rows stay interpretable.
Cross-encoders take no question and use the id "none".
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Phrasing:
    id: str
    question: str  # {query} is replaced by the query text
    frozen: bool = False


NONE = Phrasing("none", "", frozen=True)

PHRASINGS: Dict[str, Phrasing] = {
    p.id: p
    for p in [
        NONE,
        # DRAFT -- to be fixed before any test-set run (WP3).
        Phrasing("p1-relevant", "Is this passage relevant to the search query '{query}'?"),
        Phrasing("p2-answers", "Does this passage answer the question '{query}'?"),
        Phrasing("p3-useful", "Would this passage be useful to someone searching for '{query}'?"),
    ]
}


def get_phrasing(phrasing_id: str) -> Phrasing:
    try:
        return PHRASINGS[phrasing_id]
    except KeyError:
        raise KeyError(f"Unknown phrasing '{phrasing_id}'. Known: {', '.join(PHRASINGS)}") from None
