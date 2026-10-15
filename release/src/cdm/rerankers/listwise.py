"""
RankZephyr (Pradeep et al. 2023, castorini/rank_zephyr_7b_v1_full), the open listwise baseline we run ourselves
: it generates a permutation of passage identifiers, "[3] > [1] > ...", for a window of
passages and slides that window from the bottom of the candidate list to the top.

This follows rank_llm's ZephyrReranker (github.com/castorini/rank_llm, read 2026-10-08) without its vLLM backend,
with Hugging Face transformers like our other GPU rerankers:

- prompt: rank_zephyr_template.yaml -- system message, prefix, "[i] passage" per candidate, suffix -- through the
  model's chat template with the generation prompt; "[n]" inside the query and passages becomes "(n)";
- passages cut to 300 words, and further, one word per passage at a time, until the prompt plus the expected output
  fits the 4,096-token context (rank_llm cuts by the overflow / (4 x window));
- window 20, stride 10, from the bottom: 9 windows on 100 candidates; greedy decoding, at most as many new tokens as
  the full permutation "[1] > ... > [20]" takes;
- the answer's identifiers in order, duplicates and unknown ones dropped, missing ones appended in their old order.
  Not ported: ftfy's text repair (rank_llm calls fix_text on passages and prompt).

There is no probability: `scores` are descending ranks (n, n-1, ..., 1 in the final order), `probs` is None.
"""

import re
from typing import Any, Dict, List, Optional

from cdm.phrasings import Phrasing
from cdm.rerankers.base import Reranker, Scores

SYSTEM_MESSAGE = (
    "You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query"
)
PREFIX = (
    "I will provide you with {num} passages, each indicated by a numerical identifier []. Rank the passages based on "
    "their relevance to the search query: {query}.\n"
)
BODY = "[{rank}] {candidate}\n"
SUFFIX = (
    "Search Query: {query}.\nRank the {num} passages above based on their relevance to the search query. All the "
    "passages should be included and listed using identifiers, in descending order of relevance. The output format "
    "should be [] > [], e.g., [2] > [1], Answer concisely and directly and only respond with the ranking results, do "
    "not say any word or explain."
)


def replace_number(text: str) -> str:
    return re.sub(r"\[(\d+)\]", r"(\1)", text)


def parse_permutation(answer: str, n: int) -> List[int]:
    """0-based order of a window of n passages from the model's answer; unparseable parts fall back to the old order."""
    order: List[int] = []
    for m in re.findall(r"\d+", answer):
        i = int(m) - 1
        if 0 <= i < n and i not in order:
            order.append(i)
    return order + [i for i in range(n) if i not in order]


def windows(n: int, window: int, stride: int) -> List[tuple]:
    """(start, end) of the sliding windows from the bottom up, as rank_llm's sliding_windows (rank_start 0)."""
    out = []
    end, start = n, n - min(window, n)
    while end > 0 and start + stride != 0:
        start = max(start, 0)
        out.append((start, end))
        end, start = end - stride, start - stride
    return out


class RankZephyrReranker(Reranker):
    prob_kind = "none"

    def __init__(
        self,
        name: str,
        model_id: str,
        revision: Optional[str] = None,
        window: int = 20,
        stride: int = 10,
        context_size: int = 4096,
        max_passage_words: int = 300,
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        **_: Any,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.model_id = model_id
        self.revision = revision
        self.window, self.stride = window, stride
        self.context_size = context_size
        self.max_passage_words = max_passage_words
        self.dtype = dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype))
            .to(self.device)
            .eval()
        )
        self.reset_stats()

    def reset_stats(self) -> None:
        self._stats = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "incomplete_permutations": 0}

    def run_stats(self) -> Dict[str, float]:
        return dict(self._stats)

    def _output_budget(self, n: int) -> int:
        full = " > ".join(f"[{i + 1}]" for i in range(n))
        return len(self.tokenizer.encode(full, add_special_tokens=False))

    def _prompt(self, query: str, passages: List[str], max_words: int) -> List[int]:
        query = replace_number(query)
        body = "".join(
            BODY.format(rank=i + 1, candidate=replace_number(" ".join(p.strip().split()[:max_words])))
            for i, p in enumerate(passages)
        )
        user = PREFIX.format(num=len(passages), query=query) + body + SUFFIX.format(num=len(passages), query=query)
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def _fit(self, query: str, passages: List[str]) -> List[int]:
        budget = self.context_size - self._output_budget(len(passages))
        words = self.max_passage_words
        while True:
            ids = self._prompt(query, passages, words)
            if len(ids) <= budget or words <= 1:
                return ids
            words -= max(1, (len(ids) - budget) // (len(passages) * 4))

    def _rank_window(self, query: str, passages: List[str]) -> Dict[str, Any]:
        import torch

        ids = self._fit(query, passages)
        if hasattr(ids, "keys"):  # BatchEncoding (newer transformers)
            ids = ids["input_ids"]
        inputs = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            out = self.model.generate(
                inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=self._output_budget(len(passages)),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new = out[0, inputs.shape[1] :].tolist()
        answer = self.tokenizer.decode(new, skip_special_tokens=True)
        order = parse_permutation(answer, len(passages))
        named = len(set(int(m) for m in re.findall(r"\d+", answer) if 0 < int(m) <= len(passages)))
        self._stats["requests"] += 1
        self._stats["input_tokens"] += len(ids)
        self._stats["output_tokens"] += len(new)
        self._stats["incomplete_permutations"] += named < len(passages)
        return {"answer": answer, "order": order, "input_tokens": len(ids), "output_tokens": len(new)}

    def score(self, query: str, passages: List[str], phrasing: Phrasing) -> Scores:
        ranking = list(range(len(passages)))  # positions -> passage index, first-stage order to start
        records = []
        for start, end in windows(len(passages), self.window, self.stride):
            current = ranking[start:end]
            r = self._rank_window(query, [passages[i] for i in current])
            ranking[start:end] = [current[j] for j in r["order"]]
            records.append({"start": start, "end": end, **r})
        scores = [0.0] * len(passages)
        for pos, i in enumerate(ranking):
            scores[i] = float(len(passages) - pos)
        return Scores(scores=scores, probs=None, raw=records)

    def describe(self) -> Dict[str, Any]:
        d = super().describe()
        d.update(
            kind="listwise (permutation generation)",
            window=self.window,
            stride=self.stride,
            context_size=self.context_size,
            max_passage_words=self.max_passage_words,
            dtype=self.dtype,
            device=self.device,
            decoding="greedy",
            prompt_template="rank_llm rank_zephyr_template.yaml",
            model_revision=getattr(self.model.config, "_commit_hash", self.revision),
        )
        return d
