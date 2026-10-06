# Decision models as zero-shot rerankers

Can an open decision model rerank passages without retrieval training, and are its relevance
probabilities calibrated? Laya (`convaiinnovations/laya`, ModernBERT-large, 421M, Apache-2.0) answers
typed questions (noul = yes/no, score = rubric, choice) with probabilities in one forward pass. It
is the open counterpart of TypeSafe's closed Jev. This repo uses it as a pointwise reranker on TREC DL
2019/2020 and compares it with two rerankers trained on relevance data.

- **RQ1 (effectiveness).** How does zero-shot Laya compare with monoT5-base and Qwen3-Reranker-4B
  when reranking the official TREC DL passage rerank pool? Do the question type and the rule that
  turns its answer into a score matter?
- **RQ2 (calibration).** Are Laya's relevance probabilities calibrated against the human DL labels,
  raw and after its own temperature scaling, and how do they compare with the baselines'?

The paper outline is a separate doc; results are below once the runs finish.

## Setup

| Component | Choice |
| --- | --- |
| Collections | TREC DL 2019 (43 judged queries, 9,260 judged pairs) and DL 2020 (54, 11,386); MS MARCO v1 passage; grades 0-3 |
| Candidates | Official rerank-subtask pool, `msmarco-passagetest{2019,2020}-top1000` (up to 1000 per query; 41,042 + 50,024 pairs). The files are unranked, so the BM25 baseline and all tie-breaks use our bm25s scores over the pool (Lucene variant, k1 = 1.2, b = 0.75) |
| Decision model | Laya, English root checkpoint (512-token context), bf16 as shipped |
| Laya questions (`questions.py`) | **noul**: "Does the 'passage' answer the 'query'?" -> P(yes). **score**: UMBRELA's 4 levels -> P(y), y = 0-3. State = `{"query", "passage"}`. Wording fixed before any run |
| Score ranking rules | E[y] (Bayes-optimal for linear-gain nDCG), E[2^y - 1] (exponential gain), P(y >= 2) |
| Baselines | BM25; `castorini/monot5-base-msmarco` P(true); `Qwen/Qwen3-Reranker-4B` P(yes) with the model card's prompt and web-search instruction |
| Effectiveness | nDCG@10 (trec_eval linear gain; primary), nDCG@10 exponential gain, RR@10 and AP@1000 at rel >= 2, Judged@10; paired t-tests over queries, Holm-corrected |
| Calibration population | Every judged pair (independent of the pool); the judged pairs inside the pool as a second table |
| Calibration targets | rel >= 2 (primary), rel >= 1; grades 0-3 for the score distribution |
| Calibration metrics | ECE (15 equal-width bins on P(positive)), adaptive ECE, Brier, NLL, AUROC; RPS, accuracy and Spearman for the grade distribution; 95% CIs from 1,000 query-level bootstrap resamples |
| Laya conditions | raw (T = 1); shipped temperatures (noul 1.98, score 1.25); fitted with Laya's own `fit_temperature_map`, cross-fitted (fit on DL19, apply to DL20, and the reverse). Baselines raw only |

## Layout
```
src/decisionrerank/
  data.py           frozen inputs and the pair set (pool + judged)
  questions.py      the two Laya questions
  scorers.py        raw logits: Laya (its own encode/forward path), monoT5, Qwen3-Reranker-4B
  probs.py          logits -> probabilities per condition; temperatures; ranking scores
  effectiveness.py  ir_measures per-query metrics, tie-breaking, Holm
  calibration.py    ECE / aECE / Brier / NLL / AUROC / RPS, reliability bins, query bootstrap
scripts/
  prepare_data.py   freeze data/ (qrels, queries, official pool + BM25 scores, passages)
  score.py MODEL    write runs/MODEL.parquet (incremental: only unscored pairs are run)
  evaluate.py       RQ1 -> results/effectiveness.{md,csv}, per_query.csv, score_rules.csv
  calibrate.py      RQ2 -> results/calibration*.{md,csv}, figures/{reliability,grades}.{png,pdf}
tests/              metric and temperature-fitting checks
data/ runs/ logs/   not in git
```

## Reproduce
```
uv sync
mkdir -p data/raw && for y in 2019 2020; do curl -sSL -o data/raw/msmarco-passagetest$y-top1000.tsv.gz \
  https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest$y-top1000.tsv.gz; done
uv run scripts/prepare_data.py           # needs the qpp-pruning-study corpus and the rerank-depth bm25s index
for m in laya monot5 qwen3; do uv run scripts/score.py $m; done
uv run scripts/evaluate.py && uv run scripts/calibrate.py
uv run pytest -q
```

Checks built in: the Laya raw-logit path reproduces `Agent.predict_batch` probabilities under the
shipped temperatures (max abs diff 5e-5, its 4-decimal rounding); the bm25s scores reproduce the
rerank-depth top-1000 scores on every overlapping pair; no Laya state is truncated at 512 tokens.

## Results

_Filled in from results/ once the runs finish._
