"""
Set-Encoder (Schlatt et al., ECIR 2025; webis/set-encoder-*): a cross-encoder that scores a query's candidates jointly,
through lightning-ir. (Its authors call it listwise; it still outputs one score per passage.)

Each (query, passage) pair is its own sequence, as in a pointwise cross-encoder, but every sequence carries an
interaction token that the other passages of the same query attend to, so a passage's score depends on the rest of
the candidate list -- without depending on its order (permutation-invariant). That makes it the trained-reranker
counterpart of the System One models' joint mode (RQ1), next to the cross-encoders that score each pair alone.

The checkpoints were trained on, and the authors evaluate them with, BM25 top-100 lists (config.depth = 100): a list
of up to `depth` candidates is scored in one forward pass, a longer one is refused (rerank the official 1000-candidate
pools with --depth 100). For a shorter list the model fills the missing interaction tokens with samples from the
mean and std of the present ones (sample_missing_docs); the sampler is seeded per query, so runs are reproducible.

lightning-ir needs transformers < 5 and patches BERT/ELECTRA's attention internals, so this runs in its own
container (container/lightning-ir.def). fp32 by default (the authors used bf16-mixed).
Scores are the model's logits; the probability is sigmoid(logit), as for the cross-encoders, and is not calibrated.
"""

import math
import os
from typing import Any, Dict, List, Optional

from cdm.config import SEED
from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores


class SetEncoderReranker(Reranker):
    joint = True
    prob_kind = "sigmoid(logit)"

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 1,
        seed: int = SEED,
        **_: Any,
    ):
        import torch
        from huggingface_hub import snapshot_download
        from lightning_ir import CrossEncoderModule
        from tokenizers import Tokenizer

        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.seed = seed
        self.batch_size = batch_size  # queries per forward pass; 1 keeps memory flat for 100 x 256-token lists
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.path = snapshot_download(model_id, revision=revision)
        self.module = CrossEncoderModule(model_name_or_path=self.path).to(self.device).eval()
        self.config = self.module.config
        self._tokens = Tokenizer.from_file(os.path.join(self.path, "tokenizer.json"))

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        import torch

        depth = getattr(self.config, "depth", None)
        if depth is not None and len(passages) > depth:
            raise ValueError(
                f"{self.name} scores at most {depth} candidates per query (it was trained on top-{depth} "
                f"lists), got {len(passages)}: pass --depth {depth}"
            )
        torch.manual_seed(self.seed)  # sample_missing_docs draws random interaction states for short lists
        with torch.inference_mode():
            scores = self.module.score(query, list(passages)).scores.float().view(-1).cpu().tolist()
        return Scores(scores=scores, probs=[1.0 / (1.0 + math.exp(-x)) for x in scores])

    def count_truncated(self, query: str, passages: List[str]) -> int:
        """Passages longer than the model's doc_length (256 tokens for the released checkpoints)."""
        limit = getattr(self.config, "doc_length", None)
        if not limit:
            return 0
        return sum(len(e.ids) > limit for e in self._tokens.encode_batch(list(passages), add_special_tokens=False))

    def describe(self) -> Dict[str, Any]:
        from importlib import metadata

        d = super().describe()
        d.update(
            revision=self.revision,
            device=self.device,
            seed=self.seed,
            dtype="float32",
            lightning_ir_version=metadata.version("lightning-ir"),
            transformers=metadata.version("transformers"),
            **{
                k: getattr(self.config, k, None)
                for k in ("depth", "doc_length", "query_length", "sample_missing_docs", "add_extra_token")
            },
        )
        return d
