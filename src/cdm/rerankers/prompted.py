"""
The base model of a triplet (docs/intent/paper-focus.md): a backbone without any relevance training, asked S1Rank's
question zero-shot. It is the third row next to its decision fine-tune (Kev, Clef) and its reranker fine-tune
(Qwen3-Reranker, Prism), so the three differ only in training.

- mode "chat" (Option A, the main base row): the backbone's post-trained release with its chat template, thinking
  switched off; one user turn with the query, the passage and S1Rank's question without its field references.
- mode "completion" (Option B, robustness check on Qwen3-8B): the pretrained -Base checkpoint, which has no chat
  format, continues "... Answer:".

P(relevant) = softmax over the first answer token's "yes" and "no" logits, each summed (logsumexp) over the case and
leading-space variants that are single tokens. Like Qwen3-Reranker: fp32 with TF32 matmuls by default, passages cut to
fit `max_length`, nothing generated.
"""

import math
from typing import Any, Dict, List, Optional

from cdm.rerankers.pointwise_lm import _length_batches, _torch_dtype, _YesNoReranker

# S1Rank's question (src/cdm/phrasings.py, "s1rank") with "document `documents.D1`" / "the `query`" as plain words.
QUESTION = (
    "Is the document relevant to the query? Relevant means it answers the query or contains information that "
    "directly addresses it."
)
CHAT_USER = 'Query: {query}\nDocument: {doc}\n\n' + QUESTION + ' Answer with only "yes" or "no".'
COMPLETION = "Query: {query}\nDocument: {doc}\n" + QUESTION + " Answer yes or no.\nAnswer:"
DOC = "\x00DOC\x00"  # placeholder: the prompt is rendered once around it, the passage tokens go in between


class PromptedJudge(_YesNoReranker):
    prob_kind = "softmax(yes, no), zero-shot"

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: Optional[str] = None,
        mode: str = "chat",
        max_length: int = 1024,
        batch_size: int = 32,
        dtype: str = "float32",
        device: Optional[str] = None,
        **_: Any,
    ):
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        if mode not in ("chat", "completion"):
            raise ValueError(f"mode must be 'chat' or 'completion', not '{mode}'")
        super().__init__(name, model_id, revision, max_length, batch_size, dtype, device)
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Qwen3.5/3.8 releases are multimodal (…ForConditionalGeneration); text-only input works with the same class.
        arch = AutoConfig.from_pretrained(model_id, revision=revision).architectures[0]
        cls = getattr(transformers, arch, None) or transformers.AutoModelForCausalLM
        self.model = cls.from_pretrained(model_id, revision=revision, dtype=_torch_dtype(dtype)).to(self.device).eval()
        self.prefix_text, self.suffix_text = self._template().split(DOC)
        self.yes_ids = self._variants("yes")
        self.no_ids = self._variants("no")

    def _template(self) -> str:
        if self.mode == "completion":
            return COMPLETION.format(query="{query}", doc=DOC)
        messages = [{"role": "user", "content": CHAT_USER.format(query="{query}", doc=DOC)}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _variants(self, word: str) -> List[int]:
        ids = []
        for v in (word, word.capitalize(), " " + word, " " + word.capitalize()):
            t = self.tokenizer.encode(v, add_special_tokens=False)
            if len(t) == 1 and t[0] not in ids:
                ids.append(t[0])
        if not ids:
            raise ValueError(f"{self.model_id}: no single-token variant of '{word}'")
        return ids

    def _ids(self, query: str, passages: List[str]) -> List[List[int]]:
        prefix = self.tokenizer.encode(self.prefix_text.replace("{query}", query), add_special_tokens=False)
        suffix = self.tokenizer.encode(self.suffix_text, add_special_tokens=False)
        budget = max(1, self.max_length - len(prefix) - len(suffix))
        docs = self.tokenizer(passages, add_special_tokens=False, truncation=True, max_length=budget)["input_ids"]
        return [prefix + d + suffix for d in docs]

    def _log_odds(self, query: str, passages: List[str]) -> List[float]:
        import torch

        ids = self._ids(query, passages)
        out = [0.0] * len(ids)
        for idx in _length_batches([len(x) for x in ids], self.batch_size):
            enc = self.tokenizer.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                last = self.model(**enc).logits[:, -1, :].float()
                z_yes = torch.logsumexp(last[:, self.yes_ids], dim=-1)
                z_no = torch.logsumexp(last[:, self.no_ids], dim=-1)
            for i, d in zip(idx, (z_yes - z_no).cpu().tolist()):
                out[i] = d if math.isfinite(d) else 0.0
        return out

    def count_truncated(self, query: str, passages: List[str]) -> int:
        prefix = self.tokenizer.encode(self.prefix_text.replace("{query}", query), add_special_tokens=False)
        suffix = self.tokenizer.encode(self.suffix_text, add_special_tokens=False)
        budget = self.max_length - len(prefix) - len(suffix)
        docs = self.tokenizer(passages, add_special_tokens=False, truncation=False)["input_ids"]
        return sum(len(d) > budget for d in docs)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(
            mode=self.mode,
            prompt=(self.prefix_text + "{passage}" + self.suffix_text),
            yes_token_ids=self.yes_ids,
            no_token_ids=self.no_ids,
        )
        return d
