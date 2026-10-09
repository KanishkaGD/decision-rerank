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
import os
from typing import Any, Dict, List, Optional

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores

QWEN3_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the '
    'answer can only be "yes" or "no".'
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
# Prism-Qwen3.5-Reranker (Zhang 2026, arXiv:2604.23734): the same layout as Qwen3-Reranker with its own fixed system
# prompt and instruction, copied from the checkpoints' chat_template.jinja (read 2026-10-09). The card says only this
# prompt yields its calibrated score sigmoid(z_yes - z_no); the first answer token is read, nothing is generated.
PRISM_SYSTEM = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
PRISM_INSTRUCTION = (
    'Judge if the document is relevant to the query. Reply "yes" or "no".\n'
    'On "yes", also emit:\n'
    "<contribution>One sentence covering every core point the document contributes to the query, without "
    "elaboration.</contribution>\n"
    "<evidence>Self-contained rewrite of the query-relevant content. Rules:\n"
    "- Faithful: rephrase only; add or infer nothing.\n"
    "- Self-contained: evidence alone must fully answer the query.\n"
    "- Concise: drop query-irrelevant background.\n"
    "- Verbatim (no translation): proper nouns, terms, abbreviations, numbers, dates, code, URLs.\n"
    "- Output language: multilingual doc \u2192 query's language; else doc's language.</evidence>"
)


def qwen3_prefix(system: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n"


def _torch_dtype(name: str):
    import torch

    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def load_causal_lm(model_id: str, revision: Optional[str], dtype: str, **kwargs: Any):
    """
    The checkpoint's own architecture class (config.architectures[0]), else AutoModelForCausalLM. For Qwen3 and
    Prism (…ForCausalLM) both are the same class; the Qwen3.5/3.8 post-trained releases are multimodal
    (…ForConditionalGeneration) and take text-only input. The fine-tuning trainer (src/cdm/finetune.py) loads its
    base model with this too, so a LoRA adapter's module names match at scoring time.
    """
    import transformers
    from transformers import AutoConfig

    arch = (AutoConfig.from_pretrained(model_id, revision=revision).architectures or [None])[0]
    cls = getattr(transformers, arch, None) if arch else None
    cls = cls or transformers.AutoModelForCausalLM
    return cls.from_pretrained(model_id, revision=revision, dtype=_torch_dtype(dtype), **kwargs)


def qwen3_reranker_bodies(instruction: str, query: str, passages: List[str]) -> List[str]:
    return [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {p}" for p in passages]


def qwen3_reranker_ids(
    tokenizer, prefix: List[int], suffix: List[int], instruction: str, query: str, passages: List[str], max_length: int
) -> List[List[int]]:
    """
    Token ids of Qwen3-Reranker's input for each passage: system prefix, the <Instruct>/<Query>/<Document> body cut to
    fit `max_length`, and the assistant suffix after which "yes"/"no" is read. Shared by the reader and the trainer
    (src/cdm/finetune.py), so a fine-tune is trained on exactly the input it is scored on.
    """
    budget = max_length - len(prefix) - len(suffix)
    bodies = qwen3_reranker_bodies(instruction, query, passages)
    ids = tokenizer(bodies, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
    return [prefix + x + suffix for x in ids]


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
    """
    Qwen/Qwen3-Reranker-*: the model card's prompt and web-search instruction -> last-position "yes"/"no" logits. With
    `system`/`instruction` = PRISM_*, the same reader serves Prism-Qwen3.5-Reranker. With `adapter` (a LoRA adapter
    directory from `cdm finetune`, relative to CDM_ROOT or absolute), the base model `model_id` gets the adapter merged
    in: our own reranker fine-tunes (docs/notes/reranker-27b-plan.md).
    """

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 64,
        dtype: str = "float32",
        instruction: str = QWEN3_INSTRUCTION,
        system: str = QWEN3_SYSTEM,
        adapter: Optional[str] = None,
        device: Optional[str] = None,
    ):
        from transformers import AutoTokenizer

        super().__init__(name, model_id, revision, max_length, batch_size, dtype, device)
        self.instruction = instruction
        self.system = system
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        model = load_causal_lm(model_id, revision, dtype)
        self.adapter = None
        if adapter:
            from peft import PeftModel

            from cdm.config import CDM_ROOT

            self.adapter = adapter if os.path.isabs(adapter) else os.path.join(CDM_ROOT, adapter)
            if not os.path.exists(os.path.join(self.adapter, "adapter_config.json")):
                raise FileNotFoundError(f"{name}: no LoRA adapter in {self.adapter} (train it with `cdm finetune`)")
            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.prefix = self.tokenizer.encode(qwen3_prefix(system), add_special_tokens=False)
        self.suffix = self.tokenizer.encode(QWEN3_SUFFIX, add_special_tokens=False)

    def _bodies(self, query: str, passages: List[str]) -> List[str]:
        return qwen3_reranker_bodies(self.instruction, query, passages)

    def _log_odds(self, query: str, passages: List[str]) -> List[float]:
        import torch

        ids = qwen3_reranker_ids(
            self.tokenizer, self.prefix, self.suffix, self.instruction, query, passages, self.max_length
        )
        out = [0.0] * len(ids)
        for idx in _length_batches([len(x) for x in ids], self.batch_size):
            enc = self.tokenizer.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                z = self.model(**enc, logits_to_keep=1).logits[:, -1, [self.no_id, self.yes_id]].float()
            for i, (z_no, z_yes) in zip(idx, z.cpu().tolist()):
                out[i] = z_yes - z_no
        return out

    def count_truncated(self, query: str, passages: List[str]) -> int:
        budget = self.max_length - len(self.prefix) - len(self.suffix)
        ids = self.tokenizer(self._bodies(query, passages), add_special_tokens=False, truncation=False)["input_ids"]
        return sum(len(x) > budget for x in ids)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(system=self.system, instruction=self.instruction)
        if self.adapter:
            d.update(adapter=self.adapter, adapter_training=_adapter_provenance(self.adapter))
        return d


def _adapter_provenance(adapter: str) -> Dict[str, Any]:
    """The training run behind an adapter: code revision and config from the manifest `cdm finetune` wrote."""
    import json

    for path in (os.path.join(adapter, "manifest.json"), os.path.join(os.path.dirname(adapter), "manifest.json")):
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            return {"manifest": path, "code": m.get("code"), "config": m.get("config")}
    return {}
