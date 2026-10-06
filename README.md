# Decision models as zero-shot rerankers

Can a decision model rerank passages without retrieval training, and are its relevance probabilities
calibrated? Decision models answer typed questions (noul = yes/no, score = rubric, choice) with
probabilities instead of text. This repo tests two of them as pointwise rerankers on TREC DL 2019/2020:
TypeSafe's closed **Jev 1.13** (through OpenRouter) and its open counterpart **Laya**
(`convaiinnovations/laya`, ModernBERT-large, 421M, Apache-2.0). Both are compared with two rerankers
trained on relevance data.

- **RQ1 (effectiveness).** How do zero-shot decision models compare with monoT5-base and
  Qwen3-Reranker-0.6B when reranking the official TREC DL passage rerank pool? Do the question type and
  the rule that turns an answer into a score matter?
- **RQ2 (calibration).** Are their relevance probabilities calibrated against the human DL labels, as
  shipped and after temperature scaling, and how do they compare with the trained rerankers'?

## Setup

| Component | Choice |
| --- | --- |
| Collections | TREC DL 2019 (43 judged queries, 9,260 judged pairs) and DL 2020 (54, 11,386); MS MARCO v1 passage; grades 0-3 |
| Candidates | Official rerank-subtask pool, `msmarco-passagetest{2019,2020}-top1000` (up to 1000 per query; 41,042 + 50,024 pairs). The files are unranked, so the BM25 baseline and all tie-breaks use our bm25s scores over the pool (Lucene variant, k1 = 1.2, b = 0.75) |
| Decision models | **Jev 1.13** (`typesafe/jev-1.13`, resolves to `jev-1.13-20260917`; OpenRouter `/api/alpha/decisions`, one request per pair, 101,414 requests, $2.06). **Laya**, English root checkpoint (fp16 weights, 512-token context), run with autocast off and TF32 matmuls |
| Questions (`questions.py`) | Same for both models. **noul**: "Does the 'passage' answer the 'query'?" -> P(yes). **score**: UMBRELA's 4 levels -> P(y), y = 0-3. State = `{"query", "passage"}`. Wording fixed before any run |
| Score ranking rules | E[y] (Bayes-optimal for linear-gain nDCG), E[2^y - 1] (exponential gain), P(y >= 2) |
| Baselines | BM25; `castorini/monot5-base-msmarco` P(true); `Qwen/Qwen3-Reranker-0.6B` P(yes) with the model card's prompt and web-search instruction |
| Effectiveness | nDCG@10 (trec_eval linear gain; primary), nDCG@10 exponential gain, nDCG'@10 (judged passages only), RR@10 and AP@1000 at rel >= 2, Judged@10; paired t-tests over queries, Holm-corrected within each (collection, measure) |
| Calibration population | Every judged pair (independent of the pool); the judged pairs inside the pool as a second table |
| Calibration targets | rel >= 2 (primary), rel >= 1; grades 0-3 for the score distribution |
| Calibration metrics | ECE (15 equal-width bins on P(positive)), per-collection ECE, adaptive ECE, Brier, NLL, pooled and per-query AUROC; RPS, accuracy and Spearman for the grade distribution; 95% CIs and paired differences from 1,000 query-level bootstrap resamples |
| Calibration conditions | **As shipped**: every model's own probability (Laya with its checkpoint temperatures, noul 1.98, score 1.25). **Laya raw**: T = 1. **Laya fitted**: Laya's own `fit_temperature_map`. **Cross-fit T, all models**: one unclamped temperature on each model's yes/no logit (Jev: logit of its rounded P). Every fitted temperature is cross-fitted: fit on DL19, apply to DL20, and the reverse |

## Layout
```
src/decisionrerank/
  data.py           frozen inputs and the pair set (pool + judged)
  questions.py      the two questions, shared by Laya and Jev
  scorers.py        raw logits: Laya (its own encode/forward path), monoT5, Qwen3-Reranker-0.6B
  jev.py            Jev through OpenRouter, with an on-disk response cache
  probs.py          logits -> probabilities; temperatures (shipped, fitted, cross-fit); ranking scores
  effectiveness.py  ir_measures per-query metrics, exp-gain and condensed nDCG, tie-breaking, Holm
  calibration.py    ECE / aECE / Brier / NLL / AUROC / RPS, reliability bins, query bootstrap
scripts/
  prepare_data.py   freeze data/ (qrels, queries, official pool + BM25 scores, passages)
  score.py MODEL    runs/MODEL.parquet for laya | monot5 | qwen3 (checkpointed; only unscored pairs run)
  score_jev.py      runs/jev.parquet (whole queries in random order; cached; budget cap)
  evaluate.py       RQ1 -> results/effectiveness.{md,csv}, per_query.csv, score_rules.csv, tie_robustness.csv
  calibrate.py      RQ2 -> results/calibration*.{md,csv}, figures/{reliability,grades}.{png,pdf}
  sample_report.py  quick comparison on the queries Jev has fully answered (results/sample_report.md)
  check_runs.py     integrity checks -> results/checks.json
tests/              metric and temperature-fitting checks
data/ runs/ logs/   not in git; .env holds OPENROUTER_API_KEY (not in git)
```

## Reproduce
```
uv sync
mkdir -p data/raw && for y in 2019 2020; do curl -sSL -o data/raw/msmarco-passagetest$y-top1000.tsv.gz \
  https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest$y-top1000.tsv.gz; done
uv run scripts/prepare_data.py           # needs the qpp-pruning-study corpus and the rerank-depth bm25s index
for m in laya monot5 qwen3; do uv run scripts/score.py $m; done
cp .env.example .env                     # then put your OpenRouter key in .env (git-ignored)
uv run scripts/score_jev.py              # about $2
uv run scripts/evaluate.py && uv run scripts/calibrate.py && uv run scripts/check_runs.py
uv run pytest -q
```

`check_runs.py` asserts that the stored Laya logits reproduce Laya's own `Agent.predict_batch` under the
shipped temperatures on 800 pairs (max abs diff 8e-5, its 4-decimal rounding, same TF32 runtime), that no
Laya state was truncated, and that our monoT5 scores match rerank-depth's independent CPU fp32 run on
49,992 pairs (max log-odds difference 9e-5). It also records that Jev answered all 101,414 pairs with
one model version. The bm25s scores reproduce rerank-depth's stored top-1000 scores on all 48,022
overlapping pairs (asserted in `prepare_data.py`).

## Results

Zero-shot Jev reranks on par with the two trained rerankers, and its yes/no probability is the best
probability estimate of all models. Laya does not improve on BM25. Full tables:
[results/effectiveness.md](results/effectiveness.md) and [results/calibration.md](results/calibration.md).

### RQ1: effectiveness (official rerank pool, 43 + 54 queries)

| System | nDCG@10 DL19 | nDCG@10 DL20 | nDCG@10 pooled | nDCG'@10 pooled (judged only) | RR@10 pooled | Judged@10 pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.495 | 0.483 | 0.488 | 0.491 | 0.640 | 0.979 |
| monoT5-base | 0.723 | 0.714 | **0.718** | 0.721 | **0.877** | 0.987 |
| Qwen3-Reranker-0.6B | 0.695 | 0.673 | 0.682 | 0.696 | 0.802 | 0.957 |
| Jev noul: P(yes) | 0.664 | 0.698 | 0.683 | 0.741 | 0.838 | 0.857 |
| Jev noul, ties by E[y] | 0.654 | 0.689 | 0.674 | 0.738 | 0.821 | 0.847 |
| Jev score: E[y] | 0.669 | 0.716 | 0.695 | **0.751** | 0.851 | 0.868 |
| Laya noul: P(yes) | 0.518 | 0.442 | 0.476 | 0.614 | 0.679 | 0.682 |
| Laya score: E[y] | 0.488 | 0.389 | 0.433 | 0.592 | 0.618 | 0.634 |
| Laya score: E[2^y-1] | 0.485 | 0.383 | 0.429 | 0.587 | 0.611 | 0.632 |
| Laya score: P(y>=2) | 0.487 | 0.390 | 0.433 | 0.590 | 0.585 | 0.638 |

- **Jev matches the trained rerankers.** Pooled nDCG@10 of Jev score E[y] is 0.695, against 0.718 for
  monoT5-base and 0.682 for Qwen3-Reranker-0.6B. Neither difference is significant (Holm p = 1.0 for both). On
  judged passages only, Jev score is the best system (nDCG'@10 0.751), significantly above Qwen3-0.6B
  (Holm p = 0.010) and not different from monoT5 (p = 0.30). Every Jev variant is significantly above
  BM25 and Laya.
- **Jev's rounding does not drive its ranking.** Jev returns probabilities rounded to 0.01, so 91-98%
  of its pool scores tie with another passage of the same query. With ties broken at random instead of
  by BM25 (mean of 20 draws), nDCG@10 moves by at most 0.016 (score E[y]: 0.695 -> 0.693;
  [results/tie_robustness.csv](results/tie_robustness.csv)).
- **Laya does not rerank.** Every Laya variant is significantly below both trained rerankers, and every
  Jev variant is significantly above Laya noul (pooled, Holm-corrected). Laya noul does not beat BM25 on nDCG@10 (0.476 vs 0.488, p = 0.66). A third
  of its top 10 has no DL label (Judged@10 0.68). On judged passages only, it does beat BM25 (0.614 vs
  0.491), but it stays well below the other rerankers.
- **The question type matters little.** Laya's score question is no better than its noul (0.433 vs
  0.476; p = 0.017-0.027 unadjusted, 0.22-0.29 after Holm). Jev's is slightly better (0.695 vs 0.683,
  not tested separately). The ranking rule barely matters: E[y], E[2^y-1] and P(y>=2) are within 0.005
  for Laya, and temperature moves Laya's score ranking by at most 0.017 nDCG@10.

### RQ2: calibration (all 20,646 judged pairs, target rel >= 2)

The positive rate is 0.202 (DL19 0.270, DL20 0.146). "ECE coll." is the size-weighted mean of the DL19
and DL20 ECEs. The pooled ECE can hide opposite per-collection errors (Laya fitted: pooled 0.043,
per collection 0.083). Per-query AUROC averages over queries with both classes.

| System | ECE pooled [95% CI] | ECE coll. | Brier | NLL | AUROC pooled | AUROC per query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **As shipped** | | | | | | |
| monoT5-base | 0.172 [0.149, 0.195] | 0.173 | 0.178 | 1.095 | 0.773 | 0.847 |
| Qwen3-Reranker-0.6B | 0.276 [0.243, 0.310] | 0.277 | 0.276 | 1.192 | 0.764 | 0.844 |
| Laya noul (T = 1.98) | 0.056 [0.040, 0.076] | **0.064** | 0.146 | 0.453 | 0.753 | 0.811 |
| Laya score P(y>=2) (T = 1.25) | 0.168 [0.139, 0.199] | 0.169 | 0.179 | 0.540 | 0.731 | 0.793 |
| Jev noul | 0.084 [0.062, 0.111] | 0.087 | 0.127 | 0.407 | **0.858** | **0.890** |
| Jev score P(y>=2) | 0.164 [0.137, 0.192] | 0.165 | 0.166 | 1.097 | 0.857 | 0.888 |
| **Cross-fitted temperature (same fitter for all)** | | | | | | |
| monoT5-base | 0.064 [0.048, 0.094] | 0.095 | 0.142 | 0.447 | 0.748 | 0.847 |
| Qwen3-Reranker-0.6B | 0.238 [0.207, 0.272] | 0.238 | 0.202 | 0.586 | 0.760 | 0.844 |
| Laya noul | 0.043 [0.027, 0.076] | 0.083 | 0.145 | 0.454 | 0.731 | 0.811 |
| Jev noul | 0.095 [0.068, 0.123] | 0.095 | **0.124** | **0.393** | 0.857 | 0.890 |
| **Reference** | | | | | | |
| Laya noul raw (T = 1) | 0.140 [0.118, 0.161] | 0.141 | 0.166 | 0.619 | 0.753 | 0.811 |
| Base rate of the other collection | 0.124 [0.096, 0.151] | 0.124 | 0.173 | 0.538 | | 0.500 |

Paired query-bootstrap differences, all judged pairs ([results/calibration_paired.csv](results/calibration_paired.csv)):

| Comparison | Brier diff [95% CI] | NLL diff [95% CI] | ECE coll. diff [95% CI] |
| --- | ---: | ---: | ---: |
| Jev noul (shipped) - monoT5 (cross-fit) | -0.015 [-0.029, -0.001] | -0.040 [-0.082, 0.006] | -0.008 [-0.032, 0.024] |
| Jev noul (cross-fit) - monoT5 (cross-fit) | -0.018 [-0.030, -0.005] | -0.054 [-0.089, -0.019] | 0.000 [-0.029, 0.034] |
| Jev noul (shipped) - Laya noul (shipped) | -0.019 [-0.032, -0.005] | -0.046 [-0.088, -0.005] | 0.023 [-0.005, 0.046] |
| Laya noul (shipped) - monoT5 (raw) | -0.033 [-0.045, -0.020] | -0.642 [-0.813, -0.459] | -0.109 [-0.122, -0.079] |
| Laya noul (shipped) - monoT5 (cross-fit) | 0.004 [-0.003, 0.012] | 0.006 [-0.014, 0.028] | -0.031 [-0.044, -0.006] |
| Laya noul (fitted) - Laya noul (shipped) | -0.001 [-0.003, 0.001] | 0.001 [-0.010, 0.011] | 0.019 [-0.003, 0.026] |

![Reliability diagrams](figures/reliability.png)

- **As shipped, both decision models are far better calibrated than the trained rerankers.** Their ECE
  is 0.06-0.08 against 0.17-0.28. The trained rerankers push most pairs to P near 0 or 1, and at
  P near 1 only 62% (monoT5) and 42% (Qwen3) of pairs have rel >= 2. monoT5's Brier score is slightly worse
  than predicting the base rate (0.178 vs 0.173), although it ranks best.
- **Temperature scaling closes most of that gap for monoT5, but not Jev's lead.** After one cross-fitted
  temperature (T = 4.5-6.0), monoT5 matches Laya on Brier and NLL (CIs include 0). Jev stays better than
  recalibrated monoT5 on Brier as shipped, and on Brier and NLL after its own cross-fit. Its advantage is
  discrimination (per-query AUROC 0.890 vs 0.847), not ECE, where the two are tied.
  Qwen3-0.6B stays poorly calibrated even after scaling (ECE 0.238).
- **Laya's own temperature fitter adds nothing over its shipped temperature.** No paired difference
  excludes 0, and per-collection ECE is nominally worse (0.083 vs 0.064). The temperature fitted on
  DL19 (positive rate 0.270) over-predicts on DL20 (rate 0.146).
- **Raw Laya is over-confident** (ECE 0.140). Its shipped temperature fixes most of that.
- **Graded answers.** Jev's grade distribution tracks the human grade far better than Laya's (Spearman
  0.60 vs 0.38, argmax accuracy 0.40 vs 0.31), but neither beats a constant prior on RPS (0.203 and
  0.219 vs 0.160). P(y >= 2) from the score question is worse calibrated than the noul answer for both
  models. For Laya, fitting one temperature by multiclass NLL (T about 4.5) flattens the grades
  towards uniform and makes P(y >= 2) worse (ECE 0.256).

![Mean grade distribution by human grade](figures/grades.png)

### Takeaway

Jev, used zero-shot with one yes/no question per passage, reranks as well as small rerankers trained on
MS MARCO, which is in-domain for them. Its probabilities are the best relevance estimates in the study:
calibrated out of the box and the most discriminative. Laya, at 0.4B parameters, gives reasonably
calibrated probabilities, but they do not rank within a query; it is not a reranker.

### Limitations and notes
- **In-domain baselines.** monoT5 was trained on MS MARCO; DL19/20 favours it. Whether Jev or Laya saw
  MS MARCO or TREC data in training is unknown, so "zero-shot" means no retrieval-specific training on
  our side.
- **Judged-pair selection.** DL qrels were pooled from participant runs, many of them BERT-style
  rerankers, so the judged non-relevant passages are hard negatives for monoT5-like models. Calibration
  on judged pairs is conditional on that pool. About half the judged pairs lie outside the official
  rerank pool (a second table in results/calibration.md restricts to pool pairs).
- **Target choice.** Results are for rel >= 2. At rel >= 1 the ordering changes; for example, Laya's score
  P(y >= 2) as shipped has the lowest ECE. See results/calibration.md.
- **One prompt per question type,** fixed before any run; no wording sensitivity analysis.
- **Small query sets.** There are 97 queries; Holm correction is within each (collection, measure),
  not across measures.
- **Precision.** Laya and Qwen3 run with fp32 outputs and TF32 matmuls; monoT5 runs in strict fp32. Under the default bf16 autocast, 32% of Laya noul and 89% of Qwen3-0.6B pool scores tied
  within a query; the bf16 runs are kept in `runs/*_bf16.parquet`. Jev's API returns probabilities
  rounded to 0.01 (see the tie analysis above).
- **BM25 row.** The official pool is unranked; 58% of its pairs tie on bm25s score (single-term matches
  of equal length), broken by passage id. The row is our ordering of the official pool, not the
  organisers' run.
