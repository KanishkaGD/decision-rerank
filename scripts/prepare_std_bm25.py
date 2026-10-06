"""Freeze the standard BM25 run (Anserini default BM25, prebuilt msmarco-v1-passage index) for QPP.

QPP work such as QPP-GenRE predicts the quality of this run, so our QPP results on it are comparable. Produced by
    java -cp anserini-1.0.0-fatjar.jar io.anserini.search.SearchCollection -index msmarco-v1-passage \
         -topics dl{19,20}-passage -output run.dl{19,20}-passage.bm25-default.txt -bm25 -hits 1000
Writes data/bm25_std.parquet (qset, query_id, doc_id, std_rank, std_score; top 1000 of the judged queries) and
adds the texts of its top-10 passages to data/passages.parquet; those pairs join the scored pair set.
"""

import json

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from decisionrerank import data

RUNS = "/root/tools/runs/run.dl{}-passage.bm25-default.txt"
CORPUS = "/root/qpp-pruning-study/data/prepared/msmarco/corpus.parquet"


def main():
    q = data.queries()
    parts = []
    for qset, y in (("dl19", "19"), ("dl20", "20")):
        r = pd.read_csv(RUNS.format(y), sep=r"\s+", header=None, dtype=str,
                        names=["query_id", "q0", "doc_id", "std_rank", "std_score", "tag"])
        r = r[r.query_id.isin(set(q[q.qset == qset].query_id))].assign(qset=qset)
        parts.append(r[["qset", "query_id", "doc_id", "std_rank", "std_score"]])
    run = pd.concat(parts, ignore_index=True).astype({"std_rank": int, "std_score": float})
    assert run.groupby("qset").query_id.nunique().to_dict() == {"dl19": 43, "dl20": 54}
    run.to_parquet(data.DATA / "bm25_std.parquet", index=False)

    have = data.passages()
    need = sorted(set(run[run.std_rank <= 10].doc_id) - set(have.index))
    if need:
        extra = pq.read_table(CORPUS, filters=pc.field("doc_id").isin(need)).to_pandas().drop_duplicates("doc_id")
        assert len(extra) == len(need)
        pd.concat([have.reset_index(), extra], ignore_index=True).sort_values("doc_id").to_parquet(
            data.DATA / "passages.parquet", index=False)
    m = json.loads((data.DATA / "manifest.json").read_text())
    m["bm25_std"] = {"source": "Anserini 1.0.0 default BM25, prebuilt msmarco-v1-passage", "rows": len(run),
                     "top10_passages_added": len(need)}
    (data.DATA / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m["bm25_std"]))


if __name__ == "__main__":
    main()
