"""
The relevance question asked of System One models (Jev, Clef, Kev, Laya), which judge each (query, passage) pair with
a typed question rather than producing a learned relevance score.

A question refers to the state the reranker builds, not to the texts themselves: the query is the state field
`query`, and {doc} is replaced by the path of the document being judged (`documents.D3`). Questions are templates, so
call question.format(doc=...); a literal brace must be doubled.

All System One runs use S1Rank's question (docs/roadmap.md, "Setup"), fixed in advance. A different question would
be chosen on msmarco-dev only and get a new id; never edit a frozen phrasing in place, so results.csv rows stay
interpretable. Rerankers that take no question use the id "none".
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Phrasing:
    id: str
    question: str  # template: {doc} is replaced by the state path of the judged document
    frozen: bool = False


NONE = Phrasing("none", "", frozen=True)

PHRASINGS: Dict[str, Phrasing] = {
    p.id: p
    for p in [
        NONE,
        # S1Rank's Noul question, verbatim (github.com/zaesho/S1Rank, s1rank/methods.py, NOUL_INSTRUCTIONS).
        Phrasing(
            "s1rank",
            "Is document `{doc}` relevant to the `query`? Relevant means it answers the query or "
            "contains information that directly addresses it.",
            frozen=True,
        ),
    ]
}


def get_phrasing(phrasing_id: str) -> Phrasing:
    try:
        return PHRASINGS[phrasing_id]
    except KeyError:
        raise KeyError(f"Unknown phrasing '{phrasing_id}'. Known: {', '.join(PHRASINGS)}") from None
