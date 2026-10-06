# Sample comparison (2026-10-06 11:47)

97 queries Jev has fully answered (DL19 43, DL20 54), a random query sample; official rerank pool. Superscripts: Holm-corrected paired t-test p < 0.05 vs a = BM25, b = monoT5-base, c = Qwen3-Reranker-0.6B, d = Laya noul.

## Effectiveness

| System | nDCG@10 | nDCG'@10 | RR@10 | Judged@10 | ties |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.488 | 0.491 | 0.640 | 0.979 | 0.58 |
| monoT5-base | 0.718<sup>a</sup> | 0.721<sup>a</sup> | 0.877<sup>a</sup> | 0.987 | 0.00 |
| Qwen3-Reranker-0.6B | 0.682<sup>a</sup> | 0.696<sup>a</sup> | 0.802<sup>a</sup> | 0.957 | 0.00 |
| Laya noul | 0.476<sup>bc</sup> | 0.614<sup>abc</sup> | 0.679<sup>bc</sup> | 0.682 | 0.00 |
| Laya score E[y] | 0.433<sup>bc</sup> | 0.592<sup>abc</sup> | 0.618<sup>bc</sup> | 0.634 | 0.00 |
| Jev noul | 0.683<sup>ad</sup> | 0.741<sup>ad</sup> | 0.838<sup>ad</sup> | 0.857 | 0.98 |
| Jev noul, ties by E[y] | 0.674<sup>ad</sup> | 0.738<sup>ad</sup> | 0.821<sup>ad</sup> | 0.847 | 0.74 |
| Jev score E[y] | 0.695<sup>ad</sup> | 0.751<sup>acd</sup> | 0.851<sup>ad</sup> | 0.868 | 0.91 |

## Calibration (judged pairs of the sample, n = 20,646, rel >= 2 rate 0.202)

| System | ECE [95% CI] | Brier | NLL | AUROC | mean P |
| --- | ---: | ---: | ---: | ---: | ---: |
| monoT5-base | 0.172 [0.148, 0.195] | 0.178 | 1.095 | 0.773 | 0.199 |
| Qwen3-Reranker-0.6B | 0.276 [0.243, 0.311] | 0.276 | 1.192 | 0.764 | 0.410 |
| Laya noul (shipped T) | 0.056 [0.039, 0.076] | 0.146 | 0.453 | 0.753 | 0.209 |
| Laya score P(y>=2) (raw) | 0.150 [0.123, 0.180] | 0.179 | 0.543 | 0.730 | 0.349 |
| Jev noul | 0.084 [0.063, 0.110] | 0.127 | 0.407 | 0.858 | 0.275 |
| Jev score P(y>=2) | 0.164 [0.137, 0.193] | 0.166 | 1.097 | 0.857 | 0.334 |

## Score question vs human grade

| Model | RPS | Acc | Spearman E[y] |
| --- | ---: | ---: | ---: |
| Laya (raw) | 0.219 | 0.308 | 0.380 |
| Jev 1.13 | 0.203 | 0.401 | 0.601 |
