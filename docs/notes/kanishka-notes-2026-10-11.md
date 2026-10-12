# Kanishka's notes: "Decide, then Rank: negative results and to-dos for Michael" (2026-10-11)

Pasted by the owner on 2026-10-12 from Kanishka's Claude Doc. The tables are reformatted; the content is unchanged.
The comparison with our results and the open decisions are in `docs/notes/paper-handoff.md`, section 8.

Both RQs come out negative in his draft. On the same backbone the reranker fine-tune ranks better than the decision
fine-tune up to 9B, and the better calibration of decision models brings no gain in the count test. His RQ1 numbers
are ours (`paper/tables/triplets.tex`, S1Rank's question). His RQ2 numbers come from his local runs with a **new
question**.

## RQ1 (our numbers, before the kernel rerun)

Mean nDCG@10 × 100 over the ten collections:

| Backbone | Decision fine-tune | Base | Joint | Isolated | Reranker fine-tune |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | Kev-0.6B | 23.8 | 13.2 | 27.0 | 56.1 |
| Qwen3-8B | Kev-8B | 53.3 | incomplete | 52.7 | 59.4 |
| Qwen3.5-0.8B | Kev-0.8B | 37.6 | 25.3 | 35.9 | 57.1 |
| Qwen3.5-4B | Kev-4B | 52.7 | 49.8 | 51.4 | 58.7 |
| Qwen3.5-9B | Kev-9B | 53.7 | incomplete | 50.1 | 59.0 |
| Qwen3.5-9B | Clef-Flash | 53.7 | 52.8 | 54.7 | 59.0 |
| Qwen3.8-27B | Kev-27B | 57.4 | 61.1 | 59.7 | none |
| Qwen3.8-27B | Clef | 57.4 | 56.1 | 56.2 | none |

## RQ2: ECE, Brier and a count test (his local runs, new question)

In the count test, the sum of a model's probabilities over a query's candidates is its estimate of how many are
relevant. The table reports the mean absolute error against the true number, in passages.
- Scope: DL19–23, judged candidates only, relevant = grade ≥ 2, decision models in joint mode, the new question, no
  fitting.
- Reference: always predicting each year's median count gives an error of 8.6 passages.
- On the full top-100, with unjudged passages counted as non-relevant, every model over-counts by 1.4–3.5×.

| Model | Training | ECE | Brier | Count error | Estimate ÷ truth |
| --- | --- | ---: | ---: | ---: | ---: |
| Jev | decision | .073 | .126 | 6.2 | 1.31 |
| Kev-27B | decision | .077 | .130 | 6.0 | 1.31 |
| Clef-27B | decision | .092 | .139 | 6.7 | 1.38 |
| Qwen3.8-27B | base, zero-shot | .116 | .144 | 5.8 | 1.15 |
| Kev-4B | decision | .200 | .197 | 11.5 | 1.93 |
| Prism-4B | reranker | .219 | .172 | 11.5 | 2.00 |
| Qwen3-Reranker-8B | reranker | .257 | .238 | 13.0 | 2.20 |

Comparisons fixed in advance. Each is first minus second (negative = the first model is better), with 95% intervals
from 1,000 resamples of the 308 queries:

| Comparison | ECE | Brier | Count error |
| --- | --- | --- | --- |
| Kev-4B − Prism-4B (same backbone) | −.015 [−.027, −.004] | +.024 [+.017, +.032] | 0.0 [−0.6, +0.5] |
| Kev-27B − Qwen3.8-27B zero-shot | −.042 [−.057, −.024] | −.014 [−.021, −.008] | +0.3 [−0.2, +0.8] |
| Jev − Prism-4B (best of each kind) | −.152 [−.163, −.142] | −.047 [−.054, −.041] | −5.3 [−6.0, −4.7] |

His reading:
- **27B:** decision fine-tuning improves calibration over the base model.
- **4B:** mixed. Kev-4B has a lower ECE than Prism-4B but a higher Brier score.
- **Count test:** no same-backbone difference. The 27B models and Jev count better than the reranker fine-tunes,
  but the zero-shot base model does too, so this is model size, not decision fine-tuning.
- **Why the base model ties:** its errors partly cancel. Kev-27B's errors add up to +3.2 passages per query; the base
  model is −1.8 on the passages it dismisses and +3.4 on those it accepts, so +1.6 in total. Correlation with the true
  count is .79 for Kev-27B and .78 for the base model.
- **Limit of the test:** counting the passages with p ≥ 0.5 gives about the same error as summing the probabilities
  (5.8 vs 5.9 / 5.6), so the test uses little of the probabilities' detail.

## The new question (fixed 2026-10-06 in `master:src/decisionrerank/questions.py`, before any run)

- **Decision models** (a phrasing): ``Does document `{doc}` answer the `query`?``
- **Base models, chat:** `Query: {query}\nDocument: {doc}\n\nDoes the document answer the query? Answer with only "yes" or "no".`
- **Base models, completion:** `Query: {query}\nDocument: {doc}\nDoes the document answer the query? Answer yes or no.\nAnswer:`

**Why:** S1Rank's wording ("answers the query or contains information that directly addresses it") calibrates the
models to grade ≥ 1, while we score grade ≥ 2.

ECE on the judged candidates at grade ≥ 2 (local runs):

| Model | Data | ECE, S1Rank's wording | ECE, new wording |
| --- | --- | ---: | ---: |
| Jev joint | DL19–23 | .224 | .073 |
| Kev-27B joint | DL21–23 | .179 | .092 |
| Clef joint | DL21–23 | .162 | .103 |
| Qwen3.8-27B zero-shot | DL19–23 | .165 | .116 |

- Ranking barely changes. Kev-27B joint on DL21/22/23 scores 70.6 / 50.9 / 49.3 with the new wording, against
  70.0 / 51.2 / 49.2 with S1Rank's.
- He knows this breaks the rule that phrasings are chosen on msmarco-dev, and overrides it.
- His draft uses only the new wording and does not report the wording effect.

## His to-dos for Michael

1. Add the new question as a frozen phrasing (decision models) and as base-model prompts.
2. Rerun RQ1 with it: the whole triplet grid (decision fine-tunes joint and isolated, base models) on all ten
   collections. The reranker fine-tunes are unaffected.
3. Rerun RQ2 with it on DL19–23: jev-joint, kev-27b-joint, clef-joint, kev-4b-joint, qwen3.8-27b-prompted.
4. Add the count test to `cdm cutoff` (code below).
5. Regenerate results.csv and the tables.

Also missing in his draft:
- paired significance tests for RQ1
- the joint vs. isolated analysis by state length (cut to one sentence)
- the efficiency numbers
- references for the TREC DL 2021–2023 overviews

**His draft:** six pages LNCS plus references, titled "Decide, then Rank: Calibrated System One Decision Models as
Zero-Shot Rerankers". It is in Kanishka's checkout under `docs/paper/short/` and **not pushed**.

## How his local runs differ from the cluster

- Kev-4B ran in bf16. Kev-27B, Clef and Qwen3.8-27B ran in bf16, as on the cluster; the rerankers in fp32.
- transformers 5.17 with the causal-conv1d 1.7.0 kernel.
- Jev went through OpenRouter (typesafe/jev-1.13, snapshot 20260917).
- Clef splits a joint request at 16k tokens.
- His nDCG@10 matches ours within 0.5 points; Kev-27B exactly.

## The count test, as he wrote it (for `src/cdm/cutoff.py`, after `topk_metrics`)

```python
def count_test(queries: Sequence[QueryScores]) -> Dict[str, float]:
    """
    RQ2, count test: per query, the sum of the probabilities -- the model's own estimate of how many candidates are
    relevant -- against the true number of relevant candidates. count_MAE (mean absolute error, in passages) and
    count_ratio (mean estimate / mean truth) are computed on the judged candidates, the population of ECE and Brier;
    count_MAE_full / count_ratio_full on all candidates with unjudged = non-relevant (reported as a limit: models
    put probability on passages that were never judged). count_median_ref is the error of always predicting the
    collection's median count; it uses the labels and is a reference, not a baseline.
    """
    if not queries:
        return {}
    truth = np.array([q.relevant.sum() for q in queries], dtype=float)
    judged = np.array([q.probs[~q.unjudged].sum() for q in queries])
    full = np.array([q.probs.sum() for q in queries])
    mean_truth = truth.mean()
    return {
        "count_MAE": float(np.abs(judged - truth).mean()),
        "count_ratio": float(judged.mean() / mean_truth) if mean_truth > 0 else float("nan"),
        "count_MAE_full": float(np.abs(full - truth).mean()),
        "count_ratio_full": float(full.mean() / mean_truth) if mean_truth > 0 else float("nan"),
        "count_median_ref": float(np.abs(truth - np.median(truth)).mean()),
    }
```

Then, in `cutoff_analysis`, just before `return out`: `out.update(count_test(queries))`.

Add to `tests/test_cutoff.py`:

```python
def test_count_test_by_hand():
    from cdm.cutoff import count_test

    out = count_test([A, B])
    # A: judged probabilities 0.9 + 0.6 + 0.4 = 1.9 against 2 relevant; all four sum to 2.1
    # B: judged 0.4 against 0 relevant; both sum to 0.6
    assert out["count_MAE"] == pytest.approx((0.1 + 0.4) / 2)
    assert out["count_MAE_full"] == pytest.approx((0.1 + 0.6) / 2)
    assert out["count_ratio"] == pytest.approx((1.9 + 0.4) / 2)  # mean truth is (2 + 0) / 2 = 1
    assert out["count_ratio_full"] == pytest.approx((2.1 + 0.6) / 2)
    assert out["count_median_ref"] == pytest.approx(1.0)  # median count 1, errors 1 and 1
    assert count_test([]) == {}
    assert "count_MAE" in cutoff_analysis([A, B])
```

It writes five columns to `results.csv`:

| Column | Meaning |
| --- | --- |
| count_MAE | mean absolute error in passages on the judged candidates; the number in the paper |
| count_ratio | mean estimate ÷ mean truth on the judged candidates |
| count_MAE_full | the same error on all 100 candidates, unjudged = non-relevant |
| count_ratio_full | the same ratio on all 100 candidates; reported as a limit of the test |
| count_median_ref | error of always predicting the collection's median count; a reference, not a baseline |
