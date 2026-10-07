"""
results/results.csv: one row per evaluated configuration, the single table every paper table is built from.

Many Slurm jobs write to it concurrently, and each write is a read-modify-write of the whole file, so every access
goes through locked() (src/cdm/file_lock.py) and the file is replaced atomically.
"""

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Mapping

import pandas as pd

from cdm.config import RESULTS_FOLDER
from cdm.file_lock import locked as _locked_path

RESULTS_CSV_PATH = os.path.join(RESULTS_FOLDER, "results.csv")

# Columns that jointly identify a row; everything else is a metric or metadata. Not applicable -> "".
#   job          first-stage | rerank
#   system       bm25, monot5-3b, jev-joint, ...           model        Hugging Face id or API model name
#   dataset      dl19, scifact, ... (src/cdm/datasets.py)
#   first_stage  the candidate set reranked, e.g. bm25-d100 ("" for first-stage rows)
#   phrasing     question phrasing id (src/cdm/phrasings.py), "none" for models without one
#   variant      anything else that changes the result (e.g. "ft-calib-v1", "set10"); "" for the default
#   subset       "" for all queries, or e.g. "max20" for a quick run on the first 20 queries
RESULT_KEY_COLUMNS = ["job", "system", "model", "dataset", "first_stage", "phrasing", "variant", "subset"]


@contextmanager
def locked() -> Iterator[None]:
    with _locked_path(RESULTS_CSV_PATH):
        yield


def _normalize_key(key: Mapping[str, Any]) -> Dict[str, str]:
    unknown = set(key) - set(RESULT_KEY_COLUMNS)
    if unknown:
        raise ValueError(f"Unknown result key columns {sorted(unknown)}; known: {RESULT_KEY_COLUMNS}")
    return {col: "" if key.get(col) is None else str(key.get(col)) for col in RESULT_KEY_COLUMNS}


def load_results() -> pd.DataFrame:
    """The whole table, all values as strings. Call inside locked() if you also write."""
    if os.path.exists(RESULTS_CSV_PATH):
        return pd.read_csv(RESULTS_CSV_PATH, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=RESULT_KEY_COLUMNS)


def _mask(df: pd.DataFrame, key: Dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, value in key.items():
        mask &= df[col] == value
    return mask


def result_exists(key: Mapping[str, Any]) -> bool:
    with locked():
        df = load_results()
    return bool(_mask(df, _normalize_key(key)).any()) if not df.empty else False


def _write(df: pd.DataFrame) -> None:
    """Writes the table atomically, key columns first. Call inside locked()."""
    others = sorted(c for c in df.columns if c not in RESULT_KEY_COLUMNS)
    df = df[RESULT_KEY_COLUMNS + others].sort_values(RESULT_KEY_COLUMNS, kind="stable")
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    tmp = f"{RESULTS_CSV_PATH}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, RESULTS_CSV_PATH)


def rename_system(old: str, new: str) -> int:
    """
    Renames a system in every row, for a renamed registry entry (the only sanctioned way to change existing rows;
    log it in docs/experiment-log.md). Refuses if a row of `new` would collide with one of `old`. Returns the count.
    """
    with locked():
        df = load_results()
        if df.empty:
            return 0
        rows = df["system"] == old
        renamed = df[rows].assign(system=new)
        others = [c for c in RESULT_KEY_COLUMNS if c != "system"]
        clash = renamed.merge(df[df["system"] == new], on=["system"] + others)
        if not clash.empty:
            raise ValueError(f"{len(clash)} rows of '{new}' already exist with the same key; not renaming '{old}'")
        df.loc[rows, "system"] = new
        df.loc[rows, "updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write(df)
        return int(rows.sum())


def save_result(key: Mapping[str, Any], values: Mapping[str, Any]) -> None:
    """Inserts or replaces the row for `key` with `values` (metrics and metadata), plus a UTC timestamp."""
    norm = _normalize_key(key)
    row = {**norm, **{k: "" if v is None else str(v) for k, v in values.items()}}
    row["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with locked():
        df = load_results()
        if not df.empty:
            df = df[~_mask(df, norm)]
        _write(pd.concat([df, pd.DataFrame([row])], ignore_index=True).fillna(""))
