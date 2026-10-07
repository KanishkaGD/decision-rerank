"""
Rerankers trained on relevance data that read a yes/no answer from a language model's next-token logits: monoT5
(castorini, T5 fine-tuned on MS MARCO: "true"/"false") and Qwen3-Reranker (Qwen3 fine-tuned for relevance: "yes"/"no").
The supervised baselines next to the cross-encoders, as in Kanishka's study (master: src/decisionrerank/scorers.py),
whose prompts and token readout this follows exactly.

The score is the log-odds z_yes - z_no and the probability its sigmoid, i.e. the softmax over the two answer tokens,
which is how both models define P(relevant). Computed in fp32 with TF32 matmuls by default: Kanishka found that under
bf16 89% of Qwen3-Reranker-0.6B's scores tie within a query, which hands the ranking to the tie-break.
"""

import math
from typing import Any, Dict, List, Optional

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores

QWEN3_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def _length_batches(lengths: List[int], batch_size: int) -> List[List[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    return [order[s : s + batch_size] for s in range(0, len(order), batch_size)]


class _YesNoReranker(Reranker):
    prob_kind = "softmax(yes, no)"

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: Optional[str],
        max_length: int,
        batch_size: int,
        dtype: str,
        device: Optional[str],
    ):
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.max_length = max_length
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _log_odds(self, query: str, passages: List[str]) -> List[float]:
        raise NotImplementedError

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        z = self._log_odds(query, passages)
        return Scores(scores=z, probs=[1.0 / (1.0 + math.exp(-x)) for x in z])

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(
            max_length=self.max_length,
            batch_size=self.batch_size,
            dtype=self.dtype,
            tf32=True,
            device=self.device,
            model_revision=getattr(self.model.config, "_commit_hash", self.revision),
        )
        return d


class MonoT5Reranker(_YesNoReranker):
    """castorini/monot5-*: "Query: q Document: d Relevant:" -> logits of "true"/"false" at the first decoder step."""

    def __init__(
        self,
        name: str,
        model_id: str,
        tokenizer_id: str = "google-t5/t5-base",
        tokenizer_revision: Optional[str] = None,
        revision: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 128,
        dtype: str = "float32",
        device: Optional[str] = None,
    ):
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        super().__init__(name, model_id, revision, max_length, batch_size, dtype, device)
        # The tokenizer of the T5 base model, as pygaggle and Kanishka load it (same SentencePiece vocabulary).
        self.tokenizer_id = tokenizer_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(model_id, revision=revision, dtype=_torch_dtype(dtype))
            .to(self.device)
            .eval()
        )
        self.false_id, self.true_id = self.tokenizer.encode("false")[0], self.tokenizer.encode("true")[0]

    @staticmethod
    def _text(query: str, passage: str) -> str:
        return f"Query: {query} Document: {passage} Relevant:"

    def _log_odds(self, query: str, passages: List[str]) -> List[float]:
        import torch

        texts = [self._text(query, p) for p in passages]
        out = [0.0] * len(texts)
        for idx in _length_batches([len(t) for t in texts], self.batch_size):
            enc = self.tokenizer(
                [texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            start = torch.zeros((len(idx), 1), dtype=torch.long, device=self.device)
            with torch.inference_mode():
                z = self.model(**enc, decoder_input_ids=start).logits[:, 0, [self.false_id, self.true_id]].float()
            for i, (z_false, z_true) in zip(idx, z.cpu().tolist()):
                out[i] = z_true - z_false
        return out

    def count_truncated(self, query: str, passages: List[str]) -> int:
        ids = self.tokenizer([self._text(query, p) for p in passages], truncation=False)["input_ids"]
        return sum(len(x) > self.max_length for x in ids)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(tokenizer_id=self.tokenizer_id, prompt=self._text("{query}", "{passage}"))
        return d


class Qwen3Reranker(_YesNoReranker):
    """Qwen/Qwen3-Reranker-*: the model card's prompt and web-search instruction -> last-position "yes"/"no" logits."""

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 64,
        dtype: str = "float32",
        instruction: str = QWEN3_INSTRUCTION,
        device: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(name, model_id, revision, max_length, batch_size, dtype, device)
        self.instruction = instruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=_torch_dtype(dtype))
            .to(self.device)
            .eval()
        )
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = self.tokenizer.encode(QWEN3_PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(QWEN3_SUFFIX, add_special_tokens=False)

    def _bodies(self, query: str, passages: List[str]) -> List[str]:
        return [f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {p}" for p in passages]

    def _log_odds(self, query: str, passages: List[str]) -> List[float]:
        import torch

        budget = self.max_length - len(self.prefix) - len(self.suffix)
        ids = self.tokenizer(
            self._bodies(query, passages), add_special_tokens=False, truncation=True, max_length=budget
        )["input_ids"]
        ids = [self.prefix + x + self.suffix for x in ids]
        out = [0.0] * len(ids)
        for idx in _length_batches([len(x) for x in ids], self.batch_size):
            enc = self.tokenizer.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                z = self.model(**enc).logits[:, -1, [self.no_id, self.yes_id]].float()
            for i, (z_no, z_yes) in zip(idx, z.cpu().tolist()):
                out[i] = z_yes - z_no
        return out

    def count_truncated(self, query: str, passages: List[str]) -> int:
        budget = self.max_length - len(self.prefix) - len(self.suffix)
        ids = self.tokenizer(self._bodies(query, passages), add_special_tokens=False, truncation=False)["input_ids"]
        return sum(len(x) > budget for x in ids)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(instruction=self.instruction)
        return d
