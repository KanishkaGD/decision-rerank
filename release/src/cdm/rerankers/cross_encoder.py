"""Off-the-shelf cross-encoders (sentence-transformers CrossEncoder), the supervised baselines (context table of RQ1)."""

import math
from typing import Any, Dict, List, Optional

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores


class CrossEncoderReranker(Reranker):
    prob_kind = "sigmoid(logit)"

    def __init__(
        self,
        name: str,
        model_id: str,
        max_length: int = 512,
        batch_size: int = 128,
        revision: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from sentence_transformers import CrossEncoder

        self.name = name
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.model = CrossEncoder(model_id, max_length=max_length, revision=revision, device=device)
        self.tokenizer = self.model.tokenizer

    def _raw_logits(self, pairs: List[List[str]]) -> List[float]:
        import torch

        # Raw logits, whatever activation the model's config declares, so scores are comparable across models and
        # the probability is always sigmoid(logit). activation_fn is the sentence-transformers >= 4 argument name;
        # no fallback on purpose: a model default (often Sigmoid) would silently turn the logits into probabilities.
        out = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            activation_fn=torch.nn.Identity(),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [float(x) for x in out.reshape(len(pairs), -1)[:, 0]]

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        pairs = [[query, p] for p in passages]
        logits = self._raw_logits(pairs)
        probs = [1.0 / (1.0 + math.exp(-x)) for x in logits]
        return Scores(scores=logits, probs=probs)

    def count_truncated(self, query: str, passages: List[str]) -> int:
        lengths = self.tokenizer([query] * len(passages), passages, truncation=False)["input_ids"]
        return sum(len(ids) > self.max_length for ids in lengths)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(max_length=self.max_length, batch_size=self.batch_size, device=str(self.model.device))
        try:
            d["model_revision"] = self.model.model.config._commit_hash
        except AttributeError:
            pass
        return d
