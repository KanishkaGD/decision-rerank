"""
The relevance-question phrasings for System One models (Laya, Jev), which are asked a question about each
(query, passage) pair rather than producing a learned relevance score.

A question refers to the state the reranker builds, not to the texts themselves: the query is the state field
`query`, and {doc} is replaced by the path of the document being judged (Jev: `documents.D3`). Questions are
templates, so call question.format(doc=...); a literal brace must be doubled.

Protocol (research plan, "Question-phrasing protocol"): fix 3-5 phrasings in advance, select one on the dev split
only (msmarco-dev or a held-out BEIR set), report all of them. The p* texts below are a DRAFT until WP3 freezes them;
once frozen, never edit a phrasing in place -- add a new id, so results.csv rows stay interpretable.
Cross-encoders take no question and use the id "none".
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
        # S1Rank's Noul question, verbatim (github.com/zaesho/S1Rank, s1rank/methods.py, NOUL_INSTRUCTIONS), for
        # replicating its numbers (WP2). Frozen because it is fixed by that paper, not chosen by us; it is not a
        # candidate in the WP3 selection.
        Phrasing(
            "s1rank",
            "Is document `{doc}` relevant to the `query`? Relevant means it answers the query or "
            "contains information that directly addresses it.",
            frozen=True,
        ),
        # DRAFT -- to be fixed before any test-set run (WP3).
        Phrasing("p1-relevant", "Is document `{doc}` relevant to the search query `query`?"),
        Phrasing("p2-answers", "Does document `{doc}` answer the question in `query`?"),
        Phrasing("p3-useful", "Would document `{doc}` be useful to someone searching for `query`?"),
    ]
}


def get_phrasing(phrasing_id: str) -> Phrasing:
    try:
        return PHRASINGS[phrasing_id]
    except KeyError:
        raise KeyError(f"Unknown phrasing '{phrasing_id}'. Known: {', '.join(PHRASINGS)}") from None
