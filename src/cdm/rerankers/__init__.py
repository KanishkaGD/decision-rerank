"""
The reranking systems we compute, by short name (docs/roadmap.md, "Setup"). Adding one: register it in SYSTEMS, add
it to SYSTEMS in slurm/config.sh, and prefetch it. The manifest records the exact revision each run used. Baselines
taken from published papers instead of computed (RankGPT-4, RankZephyr, ...) are in results/literature.csv.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from cdm.rerankers.base import Reranker, Scores


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str  # "cross-encoder" | "monot5" | "qwen3-reranker" | "set-encoder" | "listwise" | "system-one" | "llm-judge"
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
        # newest open pointwise LLM reranker. fp32 + TF32 (Kanishka: bf16 makes Qwen3-Reranker's scores tie).
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
        # Set-Encoder (Schlatt et al., ECIR 2025): a cross-encoder that scores the passages of a query jointly --
        # each passage is its own sequence, but they exchange information through one interaction token each
        # (permutation-invariant). The trained-reranker counterpart of the System One joint mode (RQ2). Trained on
        # BM25 top-100 lists: at most 100 candidates. Runs in its own container (lightning-ir needs transformers < 5;
        # slurm/config.sh container_for).
        SystemSpec(
            "set-encoder-large",
            "set-encoder",
            "webis/set-encoder-large",
            {"batch_size": 1},
            revision="42b6e397e620db497820062433754f4eb2d4bf8b",
        ),
        # RankZephyr (listwise: generates a permutation, sliding window 20/10), run by us for DL21-23, the efficiency
        # run and as a check against the published DL19/20 numbers (docs/intent/paper-focus.md).
        SystemSpec(
            "rankzephyr-7b",
            "listwise",
            "castorini/rank_zephyr_7b_v1_full",
            revision="aa11d9da444ec3490827656c3b961d5c5f3af0eb",
        ),
        # A decoder LLM at Jev's price, read through its log-probabilities (Kanishka's reference; OpenRouter).
        SystemSpec("qwen3.7-flash", "llm-judge", "qwen/qwen3.7-flash", hf=False),
        # System One models (src/cdm/rerankers/system_one.py), all asked the same state and question, each joint and
        # isolated (_system_one; RQ2). Jev through the TypeSafe API, pinned version (no alias).
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
            for size, rev in KEV_REVISIONS.items()
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
    if kind == "listwise":
        from cdm.rerankers.listwise import RankZephyrReranker

        return RankZephyrReranker
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
