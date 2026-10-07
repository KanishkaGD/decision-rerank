"""
The reranking systems of the study, by short name (research plan, "Systems compared"). Adding one: register it in
SYSTEMS, add it to SYSTEMS in slurm/config.sh, and prefetch it. Model ids are the plan's suggestions; WP0 confirms
they exist and are current, and the manifest records the exact revision each run used.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from cdm.rerankers.base import Reranker, Scores


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str  # "cross-encoder" | "monot5" | "qwen3-reranker" | "set-encoder" | "system-one" | "llm-judge"
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
    "0.8b": "bf75a6a8848ea6960ff2ed108d9ed44c2941174f",
    "4b": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
    "9b": "db029f08b290afd9fee4aa4bbcd9ae48602d1eb0",
    "27b": "af0e6d551bdc2cc724f3e9d7a8bee1cd4fb8f7bf",
}


def _system_one(name: str, model_id: str, backend: str, revision: Optional[str] = None) -> List["SystemSpec"]:
    """
    The two input modes of a System One model, in IR terms. Both give one probability per candidate (pointwise
    output); they differ in what the model sees:
      <name>-pointwise  one candidate per request, nothing else in the state (window 1);
      <name>-listwise   all candidates of the query in one state, one Noul question each (window 0; S1Rank's "joint
                        pointwise"), split in halves only where the model's context is too short.
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
        for mode, window in (("pointwise", 1), ("listwise", 0))
    ]


SYSTEMS: Dict[str, SystemSpec] = {
    s.name: s
    for s in [
        SystemSpec("minilm-l6", "cross-encoder", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        SystemSpec("bge-m3", "cross-encoder", "BAAI/bge-reranker-v2-m3", {"batch_size": 32}),
        SystemSpec("mxbai-base", "cross-encoder", "mixedbread-ai/mxbai-rerank-base-v1", {"batch_size": 64}),
        # The supervised baselines of Kanishka's study (src/cdm/rerankers/pointwise_lm.py), plus their larger sizes,
        # which the H200 nodes run in fp32: monoT5 (T5 fine-tuned on MS MARCO) and Qwen3-Reranker.
        SystemSpec(
            "monot5-base",
            "monot5",
            "castorini/monot5-base-msmarco",
            {"tokenizer_id": T5_TOKENIZER[0], "tokenizer_revision": T5_TOKENIZER[1]},
            revision="891f24ce967761cac53e544481e7aeaf3c1fad80",
            extra_hf=(T5_TOKENIZER,),
        ),
        SystemSpec(
            "monot5-3b",
            "monot5",
            "castorini/monot5-3b-msmarco-10k",
            {"tokenizer_id": T5_TOKENIZER[0], "tokenizer_revision": T5_TOKENIZER[1], "batch_size": 32},
            revision="bc0c419a438c81f592f878ce32430a1823f5db6c",
            extra_hf=(T5_TOKENIZER,),
        ),
        SystemSpec(
            "qwen3-reranker-0.6b",
            "qwen3-reranker",
            "Qwen/Qwen3-Reranker-0.6B",
            {"batch_size": 64},
            revision="e61197ed45024b0ed8a2d74b80b4d909f1255473",
        ),
        SystemSpec(
            "qwen3-reranker-4b",
            "qwen3-reranker",
            "Qwen/Qwen3-Reranker-4B",
            {"batch_size": 32},
            revision="22e683669bc0f0bd69640a1354a6d0aebcfeede5",
        ),
        SystemSpec(
            "qwen3-reranker-8b",
            "qwen3-reranker",
            "Qwen/Qwen3-Reranker-8B",
            {"batch_size": 16},
            revision="77d193c791ed757ca307ee72715aa132723da912",
        ),
        # Set-Encoder (Schlatt et al., ECIR 2025): a listwise cross-encoder -- every passage is its own sequence, and
        # the passages of a query exchange information through one interaction token each (permutation-invariant).
        # The trained-reranker counterpart of the System One listwise mode (RQ4). Trained and evaluated on BM25
        # top-100: rerank at most 100 candidates (--depth 100 on the 1000-candidate pools). Runs in its own container
        # (lightning-ir needs transformers < 5; slurm/config.sh container_for).
        SystemSpec(
            "set-encoder-base",
            "set-encoder",
            "webis/set-encoder-base",
            {"batch_size": 1},
            revision="75749a7fe5025062f6aec0ea0129024a036de415",
        ),
        SystemSpec(
            "set-encoder-large",
            "set-encoder",
            "webis/set-encoder-large",
            {"batch_size": 1},
            revision="42b6e397e620db497820062433754f4eb2d4bf8b",
        ),
        # A decoder LLM at Jev's price, read through its log-probabilities (Kanishka's reference; OpenRouter).
        SystemSpec("qwen3.7-flash", "llm-judge", "qwen/qwen3.7-flash", hf=False),
        # System One models (src/cdm/rerankers/system_one.py), all asked the same state and questions, each in a
        # pointwise and a listwise variant (_system_one; RQ4 compares the two). Jev through the TypeSafe API, pinned
        # version (no alias).
        *_system_one("jev", "jev-1.13.0", "typesafe"),
        # Laya (open weights). Each question gets its own sequence with the state cut to ~320 tokens (English) or
        # ~768 (multilingual), so a candidate list doesn't fit: pointwise only, hence no suffix.
        SystemSpec(
            "laya",
            "system-one",
            "convaiinnovations/laya",
            {"backend": "laya", "window": 1},
            revision="7b928d828b7b0e022f929d9bd2e44165aa270148",
        ),
        SystemSpec(
            "laya-multilingual",
            "system-one",
            "convaiinnovations/laya",
            {"backend": "laya", "window": 1, "subfolder": "multilingual"},
            revision="7b928d828b7b0e022f929d9bd2e44165aa270148",
        ),
        # Cloudflare's Clef (open weights, Jev-compatible): Flash 9B fits a 48 GB GPU, the 27B model needs 80 GB.
        *_system_one("clef-flash", "Cloudflare/clef-flash", "clef", "17f0b0ad64efb65d273590632833508766b2aae6"),
        *_system_one("clef", "Cloudflare/clef", "clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
        # Kev (open weights, Jev-like; Kanishka ran the 0.8B, size-matched to Qwen3-Reranker-0.6B). 0.8B-9B fit a 48 GB
        # GPU, 27B needs 80 GB (slurm/config.sh gpu_constraint_for).
        *[
            spec
            for size, rev in KEV_REVISIONS.items()
            for spec in _system_one(f"kev-{size}", f"jaredpalmer/kev-{size}", "kev", rev)
        ],
        # Later: "laya-ft-*" (WP4), a generative listwise LLM reranker such as RankZephyr (RQ5 quality ceiling).
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
    if kind == "system-one":
        from cdm.rerankers.system_one import SystemOneReranker

        return SystemOneReranker
    if kind == "llm-judge":
        from cdm.rerankers.llm_judge import OpenRouterJudge

        return OpenRouterJudge
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
