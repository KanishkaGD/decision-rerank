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


class Reranker(ABC):
    """
    One reranking system. `joint` systems score the whole candidate set in one state (RQ4); the others score each
    (query, passage) pair on its own. `uses_phrasing` systems are asked a relevance question (src/cdm/phrasings.py).
    """

    name: str
    model_id: str
    joint: bool = False
    uses_phrasing: bool = False
    prob_kind: str = "none"  # "model" | "sigmoid(logit)" | "none"

    @abstractmethod
    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        ...

    def count_truncated(self, query: str, passages: List[str]) -> int:
        """How many (query, passage) inputs exceed the model's input length. Called outside the latency timing."""
        return 0

    def describe(self) -> Dict[str, Any]:
        """What goes into manifest.json: enough to reload exactly this model."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "joint": self.joint,
            "uses_phrasing": self.uses_phrasing,
            "prob_kind": self.prob_kind,
        }
