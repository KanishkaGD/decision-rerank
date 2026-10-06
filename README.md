# Decision models as zero-shot rerankers

Can an open decision model rerank passages without retrieval training, and are its relevance
probabilities calibrated? Laya (`convaiinnovations/laya`, ModernBERT-large, 421M, Apache-2.0) answers
typed questions (noul = yes/no, score = rubric, choice) with probabilities in one forward pass. It
is the open counterpart of TypeSafe's closed Jev. This repo uses it as a pointwise reranker on TREC DL
2019/2020 and compares it with two rerankers trained on relevance data.

- **RQ1 (effectiveness).** How does zero-shot Laya compare with monoT5-base and Qwen3-Reranker-0.6B
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
| Baselines | BM25; `castorini/monot5-base-msmarco` P(true); `Qwen/Qwen3-Reranker-0.6B` P(yes) with the model card's prompt and web-search instruction |
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
  scorers.py        raw logits: Laya (its own encode/forward path), monoT5, Qwen3-Reranker-0.6B
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

Zero-shot Laya does not rerank well, but its yes/no probability is the best calibrated of the three
models. Full tables: [results/effectiveness.md](results/effectiveness.md) and
[results/calibration.md](results/calibration.md).

### RQ1: effectiveness (official rerank pool, 43 + 54 queries)

| System | nDCG@10 DL19 | nDCG@10 DL20 | nDCG@10 pooled | nDCG'@10 pooled (judged only) | RR@10 pooled | Judged@10 pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.495 | 0.483 | 0.488 | 0.491 | 0.640 | 0.979 |
| monoT5-base | 0.723 | 0.714 | **0.718** | **0.721** | **0.877** | 0.987 |
| Qwen3-Reranker-0.6B | 0.695 | 0.673 | 0.682 | 0.696 | 0.802 | 0.957 |
| Laya noul: P(yes) | 0.518 | 0.442 | 0.476 | 0.614 | 0.679 | 0.682 |
| Laya score: E[y] | 0.488 | 0.389 | 0.433 | 0.592 | 0.618 | 0.634 |
| Laya score: E[2^y-1] | 0.485 | 0.383 | 0.429 | 0.587 | 0.611 | 0.632 |
| Laya score: P(y>=2) | 0.487 | 0.390 | 0.433 | 0.590 | 0.585 | 0.638 |

- **H1a holds, more strongly than expected.** Pooled over 97 queries, every Laya variant is significantly
  below monoT5-base and Qwen3-Reranker-0.6B on every measure (Holm-corrected paired t-tests; per
  collection, all but RR@10 on DL19). Laya noul does not beat BM25 on nDCG@10 (0.476 vs 0.488, p = 0.66).
- **Part of the gap is unjudged passages.** A third of Laya's top 10 has no DL label (Judged@10 0.68 vs
  at least 0.95 for the others). On judged passages only (nDCG'@10), Laya noul beats BM25 (0.614 vs
  0.491, p < 0.001), but it stays well below both trained rerankers (0.721, 0.696).
- **H1b fails.** The graded score question ranks worse than binary noul (0.433 vs 0.476). The ranking
  rule barely matters: E[y], E[2^y-1] and P(y>=2) are within 0.005 pooled. Temperature moves the score
  ranking by at most 0.017 nDCG@10 ([results/score_rules.csv](results/score_rules.csv)).

### RQ2: calibration (all 20,646 judged pairs, target rel >= 2, pooled; 95% query-bootstrap CIs)

| System | ECE | Brier | NLL | AUROC | mean P |
| --- | ---: | ---: | ---: | ---: | ---: |
| monoT5-base | 0.172 [0.149, 0.195] | 0.178 [0.155, 0.201] | 1.095 | **0.773** | 0.199 |
| Qwen3-Reranker-0.6B | 0.276 [0.243, 0.310] | 0.276 [0.246, 0.307] | 1.192 | 0.764 | 0.410 |
| Laya noul, raw (T = 1) | 0.140 [0.118, 0.161] | 0.166 [0.144, 0.186] | 0.619 | 0.753 | 0.155 |
| Laya noul, shipped (T = 1.98) | 0.056 [0.040, 0.076] | 0.146 [0.127, 0.162] | **0.453** | 0.753 | 0.209 |
| Laya noul, fitted (cross-fit T = 2.16 / 2.83) | **0.043** [0.027, 0.076] | **0.145** [0.128, 0.160] | 0.454 | 0.731 | 0.241 |
| Laya score P(y>=2), raw | 0.150 [0.123, 0.180] | 0.179 [0.164, 0.193] | 0.543 | 0.730 | 0.349 |
| Laya score P(y>=2), shipped (T = 1.25) | 0.168 [0.139, 0.199] | 0.179 [0.166, 0.192] | 0.540 | 0.731 | 0.369 |
| Laya score P(y>=2), fitted (T = 4.46 / 4.66) | 0.256 [0.227, 0.286] | 0.211 [0.205, 0.216] | 0.613 | 0.732 | 0.458 |
| Base rate of the other collection | 0.124 [0.096, 0.151] | 0.173 [0.152, 0.192] | 0.538 | | 0.215 |

The positive rate is 0.202 (DL19 0.270, DL20 0.146). AUROC of the pooled fitted rows mixes two temperatures.

![Reliability diagrams](figures/reliability.png)

- **H2a holds.** Raw Laya noul is over-confident (ECE 0.140). Its shipped temperature cuts ECE to 0.056,
  with a CI that does not overlap monoT5's (0.149-0.195) or Qwen3's (0.243-0.310).
- **H2b holds for noul only.** Laya's own fitter, cross-fitted on the other collection, adds little over
  the shipped temperature (0.043 pooled). On DL20 it is worse (0.102 vs 0.055): the temperature fitted on
  DL19 (positive rate 0.270) over-predicts there (mean P 0.248 vs a positive rate of 0.146).
- **H2c holds.** The trained rerankers push most pairs to P near 0 or 1 (U-shaped histograms). At P near 1
  only 62% (monoT5) and 42% (Qwen3) of pairs have rel >= 2. monoT5's Brier score is slightly worse than predicting
  the base rate (0.178 vs 0.173), although it ranks best.
- **The graded rubric carries little signal.** Laya's grade distribution has RPS 0.219 against 0.160 for a
  constant prior, argmax accuracy 0.31, and Spearman 0.38 with the human grade. Fitting one temperature by
  multiclass NLL (T near 4.5) flattens it towards uniform, which makes P(y>=2) worse (ECE 0.256).

![Mean grade distribution by human grade](figures/grades.png)

### Takeaway

Laya separates relevant from non-relevant judged passages almost as well as the trained rerankers
(AUROC 0.753 vs 0.773 / 0.764), and its probabilities are far better calibrated. But within a query's
candidate pool it ranks no better than BM25 on nDCG@10, and it promotes passages the DL pools never judged. Its
probability is a usable relevance estimate; its ranking is not competitive with even small trained
rerankers on MS MARCO, which are in-domain for them.

### Notes
- **Precision.** All models produce fp32 logits (TF32 matmuls). Under the default bf16 autocast, 32% of
  Laya noul and 89% of Qwen3-0.6B pool scores tied within a query, which handed the ranking to the
  tie-break; the bf16 runs are kept in `runs/*_bf16.parquet`.
- **BM25 row.** The official pool is unranked; 58% of its pairs tie on bm25s score (single-term matches
  of equal length), broken by passage id. The row is our ordering of the official pool, not the organisers' run.
- **monoT5 check.** Our monoT5 scores match rerank-depth's independent CPU run on 49,992 shared pairs
  (max log-odds difference 1e-4).
