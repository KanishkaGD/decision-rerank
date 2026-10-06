"""
The reranking systems of the study, by short name (research plan, "Systems compared"). Adding one: register it in
SYSTEMS, add it to SYSTEMS in slurm/config.sh, and prefetch it. Model ids are the plan's suggestions; WP0 confirms
they exist and are current, and the manifest records the exact revision each run used.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from cdm.rerankers.base import Reranker, Scores


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str  # "cross-encoder" | "system-one"
    model_id: str  # Hugging Face id (prefetched) or API model name
    params: Dict[str, Any] = field(default_factory=dict)
    hf: bool = True  # whether `cdm prefetch` should download model_id from Hugging Face
    revision: Optional[str] = None  # Hugging Face commit to pin (prefetch and loading); None = latest


SYSTEMS: Dict[str, SystemSpec] = {
    s.name: s
    for s in [
        SystemSpec("minilm-l6", "cross-encoder", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        SystemSpec("bge-m3", "cross-encoder", "BAAI/bge-reranker-v2-m3", {"batch_size": 32}),
        SystemSpec("mxbai-base", "cross-encoder", "mixedbread-ai/mxbai-rerank-base-v1", {"batch_size": 64}),
        # System One models (src/cdm/rerankers/system_one.py), all asked the same state and questions. Plain name:
        # all candidates in one request (joint, S1Rank's "joint pointwise"); "-isolated": one candidate per request.
        # Jev through the TypeSafe API, pinned version (no alias).
        SystemSpec("jev", "system-one", "jev-1.13.0", {"backend": "typesafe", "window": 0}, hf=False),
        SystemSpec("jev-isolated", "system-one", "jev-1.13.0", {"backend": "typesafe", "window": 1}, hf=False),
        # Laya (open weights). Each question gets its own sequence with the state cut to ~320 tokens (English) or
        # ~768 (multilingual), so joint scoring doesn't fit: isolated only.
        SystemSpec("laya", "system-one", "convaiinnovations/laya", {"backend": "laya", "window": 1},
                   revision="7b928d828b7b0e022f929d9bd2e44165aa270148"),
        SystemSpec("laya-multilingual", "system-one", "convaiinnovations/laya",
                   {"backend": "laya", "window": 1, "subfolder": "multilingual"},
                   revision="7b928d828b7b0e022f929d9bd2e44165aa270148"),
        # Cloudflare's Clef (open weights, Jev-compatible): Flash 9B fits a 48 GB GPU, the 27B model needs 80 GB.
        SystemSpec("clef-flash", "system-one", "Cloudflare/clef-flash", {"backend": "clef", "window": 0},
                   revision="17f0b0ad64efb65d273590632833508766b2aae6"),
        SystemSpec("clef-flash-isolated", "system-one", "Cloudflare/clef-flash", {"backend": "clef", "window": 1},
                   revision="17f0b0ad64efb65d273590632833508766b2aae6"),
        SystemSpec("clef", "system-one", "Cloudflare/clef", {"backend": "clef", "window": 0},
                   revision="2f3de3dd85f379784083b0814d997ab627200f0c"),
        SystemSpec("clef-isolated", "system-one", "Cloudflare/clef", {"backend": "clef", "window": 1},
                   revision="2f3de3dd85f379784083b0814d997ab627200f0c"),
        # Later: "laya-multilingual" (MIRACL), "laya-ft-*" (WP4), an open listwise LLM reranker such as RankZephyr
        # (RQ5 quality ceiling).
    ]
}


def _factory(kind: str) -> Callable[..., Reranker]:
    if kind == "cross-encoder":
        from cdm.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
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
