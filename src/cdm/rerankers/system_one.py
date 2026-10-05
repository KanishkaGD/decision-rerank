"""
System One decision models: Laya (open weights, the testbed) and Jev (closed API, the reference point).

Both are STUBS on purpose. The research plan's WP0 has to confirm Laya's API, checkpoints, licence and fine-tuning
notebook, and Jev access, before any code depends on them; everything known so far comes from unreviewed blog posts
(see docs/research-plan.md). What the implementations must provide, so that RQ2-RQ4 can be answered from stored outputs:

  - score() returns the model's own relevance probability per passage in Scores.probs (prob_kind = "model"),
    and Scores.scores as the ranking score (normally the same probability).
  - The primitive (Noul, Score or Choice) and the phrasing are parameters, recorded by describe().
  - joint=True variants score the whole candidate set in one state; Laya's 512/1024-token context limits the set
    size, so they take a set_size and report truncation via count_truncated().
"""

from typing import Any, Dict, List

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores


class LayaReranker(Reranker):
    prob_kind = "model"
    uses_phrasing = True

    def __init__(self, name: str, model_id: str, primitive: str = "noul", joint: bool = False, set_size: int = 0,
                 **_: Any):
        self.name = name
        self.model_id = model_id
        self.primitive = primitive
        self.joint = joint
        self.set_size = set_size
        raise NotImplementedError(
            "Laya is not wired up yet: confirm its API, checkpoints and licence first (research plan WP0), then "
            "implement LayaReranker in src/cdm/rerankers/system_one.py."
        )

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        raise NotImplementedError

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(primitive=self.primitive, set_size=self.set_size)
        return d


class JevReranker(Reranker):
    prob_kind = "model"
    uses_phrasing = True
    joint = True

    def __init__(self, name: str, model_id: str, **_: Any):
        self.name = name
        self.model_id = model_id
        raise NotImplementedError(
            "Jev needs API access (signups were paused in September, see the research plan). Once there is a key, "
            "put it in .env and implement JevReranker in src/cdm/rerankers/system_one.py; log the API cost per "
            "1k queries alongside the latency."
        )

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        raise NotImplementedError
