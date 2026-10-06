"""Raw per-pair outputs of each model; probabilities are derived later (``probs.py``).

Every scorer takes a pairs frame (``query``, ``passage``) and returns a frame on the same index
holding raw logits, so temperatures and other transforms can be applied offline. All models produce
fp32 logits (TF32 matmuls, fp32 accumulation and outputs): under bf16, Laya's noul and Qwen3's yes/no logits are coarse enough that 32% and 89% of a
query's pool scores tie (runs/*_bf16.parquet), which hands the ranking to the tie-break.

- laya:   noul_z0, noul_z1 (false, true) and score_z0..score_z3, the per-option logits that
          Laya's decoder divides by temperature; noul_trunc / score_trunc flag a clipped state.
- monot5: z_false, z_true, the decoder logits of "false"/"true" (castorini/monot5-base-msmarco).
- qwen3:  z_no, z_yes, the last-position logits of "no"/"yes" with the model card's prompt
          and default web-search instruction (Qwen/Qwen3-Reranker-0.6B).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from laya.common import _reuse_question_tokens

torch.backends.cuda.matmul.allow_tf32 = True  # fp32 outputs at near-bf16 speed on Ampere
torch.backends.cudnn.allow_tf32 = True

from .questions import QUESTIONS, state

LAYA_ID = "convaiinnovations/laya"
MONOT5_ID, MONOT5_TOK = "castorini/monot5-base-msmarco", "t5-base"
QWEN3_ID = "Qwen/Qwen3-Reranker-0.6B"
QWEN3_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
QWEN3_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
QWEN3_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _length_batches(lengths: np.ndarray, batch_size: int):
    order = np.argsort(lengths, kind="stable")
    for s in range(0, len(order), batch_size):
        yield order[s:s + batch_size]


def load_laya(device: str = "cuda"):
    import laya

    agent = laya.load(LAYA_ID, device=device)
    agent.amp_enabled = False  # fp32 weights; Laya autocasts to bf16 on CUDA by default
    return agent


@torch.no_grad()
@_reuse_question_tokens
def laya_logits(pairs: pd.DataFrame, agent=None, batch_size: int = 256) -> pd.DataFrame:
    """The raw option logits `predict_batch` would decode, one forward pass per batch of states."""
    from laya.agent import _option_logits
    from laya.common import collate_items

    agent = agent or load_laya()
    ids = list(QUESTIONS)
    internal = {qid: agent._to_internal(QUESTIONS[qid]) for qid in ids}
    width = {qid: len(QUESTIONS[qid]["criteria"]) if "criteria" in QUESTIONS[qid] else 2 for qid in ids}
    states = [state(q, d) for q, d in zip(pairs["query"], pairs["passage"])]
    out = {f"{qid}_z{j}": np.full(len(states), np.nan, np.float32) for qid in ids for j in range(width[qid])}
    trunc = {f"{qid}_trunc": np.zeros(len(states), bool) for qid in ids}
    for idx in _length_batches(pairs["passage"].str.len().to_numpy(), batch_size):
        encoded = [agent._encode_state(states[i], ids, internal) for i in idx]
        assert all(len(items) == len(ids) for items in encoded), "a state spilled into extra windows"
        logits, _act = agent._forward(collate_items(encoded, agent.tok.pad_token_id))  # numpy
        for b, (i, items) in enumerate(zip(idx, encoded)):
            rows = _option_logits(logits, items, b * len(ids))
            for qid, row, item in zip(ids, rows, items):
                assert len(row) == width[qid]
                for j in range(width[qid]):
                    out[f"{qid}_z{j}"][i] = row[j]
                trunc[f"{qid}_trunc"][i] = item["state_stats"]["truncated"]
    return pd.DataFrame({**out, **trunc}, index=pairs.index)


def load_monot5(device: str = "cuda"):
    from transformers import AutoTokenizer, T5ForConditionalGeneration

    return (AutoTokenizer.from_pretrained(MONOT5_TOK),
            T5ForConditionalGeneration.from_pretrained(MONOT5_ID).to(device).eval())


@torch.no_grad()
def monot5_logits(pairs: pd.DataFrame, loaded=None, batch_size: int = 128, device: str = "cuda") -> pd.DataFrame:
    tok, model = loaded or load_monot5(device)
    tf = [tok.encode("false")[0], tok.encode("true")[0]]
    text = [f"Query: {q} Document: {d} Relevant:" for q, d in zip(pairs["query"], pairs["passage"])]
    z = np.empty((len(text), 2), np.float32)
    for idx in _length_batches(np.array([len(t) for t in text]), batch_size):
        enc = tok([text[i] for i in idx], padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(device)
        dec = torch.zeros((len(idx), 1), dtype=torch.long, device=device)
        z[idx] = model(**enc, decoder_input_ids=dec).logits[:, 0, tf].float().cpu().numpy()
    return pd.DataFrame({"z_false": z[:, 0], "z_true": z[:, 1]}, index=pairs.index)


def load_qwen3(device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return (AutoTokenizer.from_pretrained(QWEN3_ID, padding_side="left"),
            AutoModelForCausalLM.from_pretrained(QWEN3_ID, dtype=torch.float32).to(device).eval())


@torch.no_grad()
def qwen3_logits(pairs: pd.DataFrame, loaded=None, batch_size: int = 64, device: str = "cuda",
                 max_tokens: int = 1024) -> pd.DataFrame:
    tok, model = loaded or load_qwen3(device)
    no, yes = tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("yes")
    pre = tok.encode(QWEN3_PREFIX, add_special_tokens=False)
    suf = tok.encode(QWEN3_SUFFIX, add_special_tokens=False)
    body = [f"<Instruct>: {QWEN3_INSTRUCTION}\n<Query>: {q}\n<Document>: {d}"
            for q, d in zip(pairs["query"], pairs["passage"])]
    ids = tok(body, add_special_tokens=False, truncation=True,
              max_length=max_tokens - len(pre) - len(suf))["input_ids"]
    ids = [pre + x + suf for x in ids]
    z = np.empty((len(ids), 2), np.float32)
    for idx in _length_batches(np.array([len(x) for x in ids]), batch_size):
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(device)
        z[idx] = model(**enc).logits[:, -1, [no, yes]].float().cpu().numpy()
    return pd.DataFrame({"z_no": z[:, 0], "z_yes": z[:, 1]}, index=pairs.index)


SCORERS = {"laya": (load_laya, laya_logits), "monot5": (load_monot5, monot5_logits),
           "qwen3": (load_qwen3, qwen3_logits)}  # name -> (load once, score(pairs, loaded))
