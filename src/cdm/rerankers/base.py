from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cdm.phrasings import Phrasing


@dataclass
class Scores:
    """A reranker's output for one query's candidate list, aligned with the passages it was given."""

    scores: List[float]  # ranking scores (raw: logits for cross-encoders)
    # Probability of relevance, if the system defines one. For System One models this is the model's own (calibrated)
    # decision probability -- the quantity RQ2/RQ3 study; for cross-encoders, sigmoid(logit), the baseline it is
    # compared against. prob_kind in describe() says which.
    probs: Optional[List[float]] = None
    # Raw model outputs that don't fit one number per passage, e.g. one record per API request (Jev). The rerank CLI
    # writes them, with the qid, to outputs/.../responses.jsonl.gz.
    raw: Optional[List[Dict[str, Any]]] = None


class Reranker(ABC):
    """
    One reranking system. Every system outputs one score per candidate; `listwise` systems compute it with the whole
    candidate list in view (System One listwise mode, Set-Encoder; RQ4), pointwise ones from each (query, passage)
    pair alone. `uses_phrasing` systems are asked a relevance question (src/cdm/phrasings.py).
    """

    name: str
    model_id: str
    listwise: bool = False
    uses_phrasing: bool = False
    prob_kind: str = "none"  # "model" | "sigmoid(logit)" | "none"
    warmup: bool = True  # whether the rerank CLI scores one query untimed first (CUDA kernels, caches)

    @abstractmethod
    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores: ...

    def count_truncated(self, query: str, passages: List[str]) -> int:
        """How many (query, passage) inputs exceed the model's input length. Called outside the latency timing."""
        return 0

    def reset_stats(self) -> None:
        """Starts a new count for run_stats() (called once per phrasing)."""

    def run_stats(self) -> Dict[str, float]:
        """Totals since reset_stats() that go into results.csv, e.g. API requests, tokens and cost."""
        return {}

    def describe(self) -> Dict[str, Any]:
        """What goes into manifest.json: enough to reload exactly this model."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "listwise": self.listwise,
            "uses_phrasing": self.uses_phrasing,
            "prob_kind": self.prob_kind,
        }
