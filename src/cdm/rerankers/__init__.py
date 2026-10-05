"""
The reranking systems of the study, by short name (research plan, "Systems compared"). Adding one: register it in
SYSTEMS, add it to SYSTEMS in slurm/config.sh, and prefetch it. Model ids are the plan's suggestions; WP0 confirms
they exist and are current, and the manifest records the exact revision each run used.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

from cdm.rerankers.base import Reranker, Scores


@dataclass(frozen=True)
class SystemSpec:
    name: str
    kind: str  # "cross-encoder" | "laya" | "jev"
    model_id: str  # Hugging Face id (prefetched) or API model name
    params: Dict[str, Any] = field(default_factory=dict)
    hf: bool = True  # whether `cdm prefetch` should download model_id from Hugging Face


SYSTEMS: Dict[str, SystemSpec] = {
    s.name: s
    for s in [
        SystemSpec("minilm-l6", "cross-encoder", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        SystemSpec("bge-m3", "cross-encoder", "BAAI/bge-reranker-v2-m3", {"batch_size": 32}),
        SystemSpec("mxbai-base", "cross-encoder", "mixedbread-ai/mxbai-rerank-base-v1", {"batch_size": 64}),
        # Placeholders until WP0: the Hugging Face id and primitive of Laya's English checkpoint, Jev's API name.
        SystemSpec("laya", "laya", "TODO/laya-english", {"primitive": "noul"}, hf=False),
        SystemSpec("jev", "jev", "jev-1.13.0", hf=False),
        # Later: "laya-multilingual" (MIRACL), "laya-ft-*" (WP4), an open listwise LLM reranker such as RankZephyr
        # (RQ5 quality ceiling).
    ]
}


def _factory(kind: str) -> Callable[..., Reranker]:
    if kind == "cross-encoder":
        from cdm.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker
    if kind == "laya":
        from cdm.rerankers.system_one import LayaReranker

        return LayaReranker
    if kind == "jev":
        from cdm.rerankers.system_one import JevReranker

        return JevReranker
    raise ValueError(f"Unknown reranker kind '{kind}'")


def get_system(name: str) -> SystemSpec:
    try:
        return SYSTEMS[name]
    except KeyError:
        raise KeyError(f"Unknown system '{name}'. Known: {', '.join(SYSTEMS)}") from None


def load_reranker(name: str, **overrides: Any) -> Reranker:
    spec = get_system(name)
    params = {**spec.params, **{k: v for k, v in overrides.items() if v is not None}}
    return _factory(spec.kind)(name=spec.name, model_id=spec.model_id, **params)


__all__ = ["Reranker", "Scores", "SystemSpec", "SYSTEMS", "get_system", "load_reranker"]
