"""
The reranking systems of the paper, by short name. Every Hugging Face model is pinned to a commit; the manifest of
each run records it. Baselines taken from published papers instead of computed (RankGPT-4, and RankZephyr on DL19/20)
are in results/literature.csv.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from cdm.rerankers.base import Reranker, Scores
from cdm.rerankers.pointwise_lm import PRISM_INSTRUCTION, PRISM_SYSTEM


@dataclass(frozen=True)
class SystemSpec:
    name: str
    # cross-encoder | monot5 | qwen3-reranker | set-encoder | listwise | prompted | system-one
    kind: str
    model_id: str  # Hugging Face id (prefetched) or API model name
    params: Dict[str, Any] = field(default_factory=dict)
    hf: bool = True  # whether `cdm prefetch` should download model_id from Hugging Face
    revision: Optional[str] = None  # Hugging Face commit to pin (prefetch and loading); None = latest
    # Further Hugging Face repos the model loads, (id, revision), for `cdm prefetch` (Kev's base models are found
    # from the checkpoint itself).
    extra_hf: Tuple[Tuple[str, Optional[str]], ...] = ()


T5_TOKENIZER = ("google-t5/t5-base", "a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1")
# Kev 1.0 checkpoints (github.com/jaredpalmer/kev, Hugging Face commits read 2026-10-07).
KEV_REVISIONS = {
    "4b": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
    "27b": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
}
# The rest of the triplets: Kev on Qwen3-0.6B/8B-Base (the bases of
# Qwen3-Reranker-0.6B/8B) and on Qwen3.5-0.8B/9B-Base (next to Prism), read 2026-10-09.
KEV_SCALE_REVISIONS = {
    "0.6b": "dece6dba8d43f0f7ded45e9f5b9df12474d90843",
    "8b": "c80773da7f383f93c4dbff0c0b008e0463f9145a",
    "0.8b": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
    "9b": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
}
# Base rows of the triplets: the post-trained release, asked zero-shot. (name, Hugging Face id, revision, dtype,
# mode)
PROMPTED = [
    ("qwen3-0.6b-prompted", "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", "float32", "chat"),
    ("qwen3-8b-prompted", "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "float32", "chat"),
    ("qwen3.5-0.8b-prompted", "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17", "float32", "chat"),
    ("qwen3.5-4b-prompted", "Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "float32", "chat"),
    ("qwen3.5-9b-prompted", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "float32", "chat"),
    # 27B in fp32 would be ~108 GB: bf16, the precision Kev-27B and Clef run in.
    ("qwen3.8-27b-prompted", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "bfloat16", "chat"),
]
# Prism-Qwen3.5-Reranker (Zhang 2026): reranker fine-tunes of Qwen3.5, read like Qwen3-Reranker with Prism's prompt.
PRISM_REVISIONS = {
    "0.8b": "c60729783cb2f0662c0b054a02964402a211e3f4",
    "4b": "bfa8c64ffd688abd4a4dc2c724dd1d83a60a6e92",
    "9b": "1c40be94424cde2e8a3be44487492e63861e1601",
}

# Our own reranker fine-tunes (`cdm finetune-data`, `cdm finetune`; Ours-27B and Ours-9B in the paper): a LoRA
# adapter on the post-trained release, trained with Qwen3-Reranker's prompt and read like it. The 27B one fills the
# reranker cell of the Qwen3.8-27B triplet (no reranker fine-tune of it is released); the 9B one validates the recipe
# against Prism-9B on the same backbone. (name, base, revision, dtype at scoring time, batch size)
OWN_RERANKERS = [
    ("qwen3.8-27b-reranker", "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", "bfloat16", 16),
    ("qwen3.5-9b-reranker", "Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "float32", 16),
]
# The checkpoint each fine-tune is scored with, chosen on msmarco-dev by the rule fixed in the plan (keep `final`
# unless an earlier checkpoint is more than 1 point better on MRR@10): checkpoint-800 for the 9B one, `final` for the
# 27B one.
SELECTED_CHECKPOINT = {"qwen3.5-9b-reranker": "checkpoint-800"}


def _system_one(name: str, model_id: str, backend: str, revision: Optional[str] = None) -> List["SystemSpec"]:
    """
    The two modes of a System One model, in S1Rank's terms. Both are pointwise -- one Noul probability per candidate,
    ranked by that probability -- and differ only in what the model sees:
      <name>-isolated  one candidate per request, nothing else in the state (window 1);
      <name>-joint     all candidates of the query in one state, one question each (window 0), split in halves only
                       where the model's context is too short.
    """
    return [
        SystemSpec(
            f"{name}-{mode}",
            "system-one",
            model_id,
            {"backend": backend, "window": window},
            revision=revision,
            hf=backend != "typesafe",
        )
        for mode, window in (("isolated", 1), ("joint", 0))
    ]


SYSTEMS: Dict[str, SystemSpec] = {
    s.name: s
    for s in [
        # Trained rerankers. bge-reranker-v2-m3 and monoT5-3B are S1Rank's baselines; Qwen3-Reranker-8B is the
        # newest open pointwise LLM reranker. fp32 + TF32 (bf16 makes Qwen3-Reranker's scores tie).
        SystemSpec("bge-m3", "cross-encoder", "BAAI/bge-reranker-v2-m3", {"batch_size": 32}),
        SystemSpec(
            "monot5-3b",
            "monot5",
            "castorini/monot5-3b-msmarco-10k",
            {"tokenizer_id": T5_TOKENIZER[0], "tokenizer_revision": T5_TOKENIZER[1], "batch_size": 32},
            revision="bc0c419a438c81f592f878ce32430a1823f5db6c",
            extra_hf=(T5_TOKENIZER,),
        ),
        SystemSpec(
            "qwen3-reranker-8b",
            "qwen3-reranker",
            "Qwen/Qwen3-Reranker-8B",
            {"batch_size": 16},
            revision="77d193c791ed757ca307ee72715aa132723da912",
        ),
        # Reranker fine-tunes of the triplets: Qwen3-Reranker-0.6B and Prism next to Kev and the base.
        SystemSpec(
            "qwen3-reranker-0.6b",
            "qwen3-reranker",
            "Qwen/Qwen3-Reranker-0.6B",
            {"batch_size": 64},
            revision="e61197ed45024b0ed8a2d74b80b4d909f1255473",
        ),
        *[
            SystemSpec(
                f"prism-{size}",
                "qwen3-reranker",
                f"infgrad/Prism-Qwen3.5-Reranker-{size.upper()}",
                {"batch_size": 16, "system": PRISM_SYSTEM, "instruction": PRISM_INSTRUCTION},
                revision=rev,
            )
            for size, rev in PRISM_REVISIONS.items()
        ],
        *[
            SystemSpec(name, "prompted", model_id, {"dtype": dtype, "mode": mode, "batch_size": 16}, revision=rev)
            for name, model_id, rev, dtype, mode in PROMPTED
        ],
        # The base models asked the paper's question ("answer"; the decision models get it as a phrasing): the systems
        # <name>-answer. The ones above ask S1Rank's question (question ablation).
        *[
            SystemSpec(
                f"{name}-answer",
                "prompted",
                model_id,
                {"dtype": dtype, "mode": mode, "question": "answer", "batch_size": 16},
                revision=rev,
            )
            for name, model_id, rev, dtype, mode in PROMPTED
            if mode == "chat"
        ],
        *[
            SystemSpec(
                name,
                "qwen3-reranker",
                model_id,
                {
                    "batch_size": batch,
                    "dtype": dtype,
                    "adapter": f"checkpoints/{name}/{SELECTED_CHECKPOINT.get(name, 'final')}",
                },
                revision=rev,
            )
            for name, model_id, rev, dtype, batch in OWN_RERANKERS
        ],
        # Set-Encoder (Schlatt et al., ECIR 2025): a cross-encoder that scores the passages of a query jointly --
        # each passage is its own sequence, but they exchange information through one interaction token each
        # (permutation-invariant). The trained-reranker counterpart of the System One joint mode (RQ1). Trained on
        # BM25 top-100 lists: at most 100 candidates. Runs in its own container (lightning-ir needs transformers < 5;
        # container/lightning-ir.def).
        SystemSpec(
            "set-encoder-large",
            "set-encoder",
            "webis/set-encoder-large",
            {"batch_size": 1},
            revision="42b6e397e620db497820062433754f4eb2d4bf8b",
        ),
        # RankZephyr (listwise: generates a permutation, sliding window 20/10), run by us for DL21-23 and as a check
        # against the published DL19/20 numbers.
        SystemSpec(
            "rankzephyr-7b",
            "listwise",
            "castorini/rank_zephyr_7b_v1_full",
            revision="aa11d9da444ec3490827656c3b961d5c5f3af0eb",
        ),
        # System One models (src/cdm/rerankers/system_one.py), all asked the same state and question, each joint and
        # isolated (_system_one; RQ1). Jev through the TypeSafe API, pinned version (no alias).
        *_system_one("jev", "jev-1.13.0", "typesafe"),
        # Laya (open weights). Each question gets its own sequence with the state cut to ~320 tokens, so a candidate
        # list doesn't fit: isolated only, hence no suffix.
        SystemSpec(
            "laya",
            "system-one",
            "convaiinnovations/laya",
            {"backend": "laya", "window": 1},
            revision="7b928d828b7b0e022f929d9bd2e44165aa270148",
        ),
        # Cloudflare's Clef (open weights, Jev-compatible): Flash 9B fits a 48 GB GPU, the 27B model needs 80 GB.
        *_system_one("clef-flash", "Cloudflare/clef-flash", "clef", "17f0b0ad64efb65d273590632833508766b2aae6"),
        *_system_one("clef", "Cloudflare/clef", "clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
        # Kev (open weights, Jev-like LoRA adapters on Qwen3.5), two sizes: 4B fits a 48 GB GPU, 27B needs 80 GB.
        *[
            spec
            for size, rev in {**KEV_REVISIONS, **KEV_SCALE_REVISIONS}.items()
            for spec in _system_one(f"kev-{size}", f"jaredpalmer/kev-{size}", "kev", rev)
        ],
    ]
}


def _factory(kind: str) -> Callable[..., Reranker]:
    if kind == "cross-encoder":
        from cdm.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
    if kind == "monot5":
        from cdm.rerankers.pointwise_lm import MonoT5Reranker

        return MonoT5Reranker
    if kind == "qwen3-reranker":
        from cdm.rerankers.pointwise_lm import Qwen3Reranker

        return Qwen3Reranker
    if kind == "set-encoder":
        from cdm.rerankers.set_encoder import SetEncoderReranker

        return SetEncoderReranker
    if kind == "prompted":
        from cdm.rerankers.prompted import PromptedJudge

        return PromptedJudge
    if kind == "listwise":
        from cdm.rerankers.listwise import RankZephyrReranker

        return RankZephyrReranker
    if kind == "system-one":
        from cdm.rerankers.system_one import SystemOneReranker

        return SystemOneReranker
    raise ValueError(f"Unknown reranker kind '{kind}'")


def get_system(name: str) -> SystemSpec:
    try:
        return SYSTEMS[name]
    except KeyError:
        raise KeyError(f"Unknown system '{name}'. Known: {', '.join(SYSTEMS)}") from None


def load_reranker(name: str, **overrides: Any) -> Reranker:
    spec = get_system(name)
    params = {**spec.params, **{k: v for k, v in overrides.items() if v is not None}}
    if spec.revision:
        params.setdefault("revision", spec.revision)
    return _factory(spec.kind)(name=spec.name, model_id=spec.model_id, **params)


__all__ = ["Reranker", "Scores", "SystemSpec", "SYSTEMS", "get_system", "load_reranker"]
