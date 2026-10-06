# Effectiveness (official TREC DL rerank pool, up to 1000 per query)

Superscripts: significantly different (paired t-test, Holm-corrected p < 0.05) from a = BM25, b = monoT5-base, c = Qwen3-Reranker-0.6B, d = Laya noul.

## DL19

| System | nDCG@10 | nDCG@10 (exp) | nDCG'@10 | RR@10 | AP@1000 | Judged@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.495 | 0.429 | 0.496 | 0.662 | 0.274 | 0.991 |
| monoT5-base | 0.723<sup>a</sup> | 0.660<sup>a</sup> | 0.729<sup>a</sup> | 0.890 | 0.457<sup>a</sup> | 0.974 |
| Qwen3-Reranker-0.6B | 0.695<sup>a</sup> | 0.630<sup>a</sup> | 0.707<sup>a</sup> | 0.813 | 0.446<sup>a</sup> | 0.949 |
| Laya noul: P(yes) | 0.518 | 0.469 | 0.646<sup>a</sup> | 0.748 | 0.323 | 0.700 |
| Laya score: E[y] | 0.488<sup>bc</sup> | 0.432<sup>bc</sup> | 0.642<sup>ab</sup> | 0.671<sup>b</sup> | 0.296<sup>bc</sup> | 0.653 |
| Laya score: E[2^y-1] | 0.485<sup>bc</sup> | 0.430<sup>bc</sup> | 0.633<sup>ab</sup> | 0.664<sup>b</sup> | 0.291<sup>bc</sup> | 0.651 |
| Laya score: P(y>=2) | 0.487<sup>bc</sup> | 0.433<sup>bc</sup> | 0.641<sup>ab</sup> | 0.673<sup>b</sup> | 0.296<sup>bc</sup> | 0.647 |
| Jev noul: P(yes) | 0.664<sup>ad</sup> | 0.606<sup>ad</sup> | 0.722<sup>ad</sup> | 0.852 | 0.465<sup>ad</sup> | 0.849 |
| Jev noul, ties by E[y] | 0.654<sup>abd</sup> | 0.592<sup>ad</sup> | 0.720<sup>ad</sup> | 0.830 | 0.463<sup>ad</sup> | 0.842 |
| Jev score: E[y] | 0.669<sup>ad</sup> | 0.612<sup>ad</sup> | 0.728<sup>ad</sup> | 0.845 | 0.470<sup>ad</sup> | 0.853 |

## DL20

| System | nDCG@10 | nDCG@10 (exp) | nDCG'@10 | RR@10 | AP@1000 | Judged@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.483 | 0.441 | 0.487 | 0.623 | 0.268 | 0.970 |
| monoT5-base | 0.714<sup>a</sup> | 0.678<sup>a</sup> | 0.716<sup>a</sup> | 0.866<sup>a</sup> | 0.478<sup>a</sup> | 0.996 |
| Qwen3-Reranker-0.6B | 0.673<sup>a</sup> | 0.627<sup>a</sup> | 0.688<sup>a</sup> | 0.794<sup>a</sup> | 0.450<sup>a</sup> | 0.963 |
| Laya noul: P(yes) | 0.442 | 0.413 | 0.588 | 0.625 | 0.276 | 0.669 |
| Laya score: E[y] | 0.389<sup>bc</sup> | 0.355<sup>bc</sup> | 0.552<sup>bc</sup> | 0.575<sup>bc</sup> | 0.236<sup>bc</sup> | 0.619 |
| Laya score: E[2^y-1] | 0.383<sup>bc</sup> | 0.351<sup>bc</sup> | 0.550<sup>bc</sup> | 0.568<sup>bc</sup> | 0.229<sup>bc</sup> | 0.617 |
| Laya score: P(y>=2) | 0.390<sup>bc</sup> | 0.354<sup>bc</sup> | 0.550<sup>bc</sup> | 0.515<sup>bc</sup> | 0.237<sup>bc</sup> | 0.631 |
| Jev noul: P(yes) | 0.698<sup>ad</sup> | 0.661<sup>ad</sup> | 0.756<sup>ad</sup> | 0.826<sup>ad</sup> | 0.484<sup>ad</sup> | 0.863 |
| Jev noul, ties by E[y] | 0.689<sup>ad</sup> | 0.650<sup>ad</sup> | 0.752<sup>ad</sup> | 0.813<sup>ad</sup> | 0.482<sup>ad</sup> | 0.852 |
| Jev score: E[y] | 0.716<sup>ad</sup> | 0.680<sup>ad</sup> | 0.770<sup>acd</sup> | 0.855<sup>ad</sup> | 0.506<sup>ad</sup> | 0.880 |

## DL19+20

| System | nDCG@10 | nDCG@10 (exp) | nDCG'@10 | RR@10 | AP@1000 | Judged@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.488 | 0.436 | 0.491 | 0.640 | 0.271 | 0.979 |
| monoT5-base | 0.718<sup>a</sup> | 0.670<sup>a</sup> | 0.721<sup>a</sup> | 0.877<sup>a</sup> | 0.469<sup>a</sup> | 0.987 |
| Qwen3-Reranker-0.6B | 0.682<sup>a</sup> | 0.628<sup>a</sup> | 0.696<sup>a</sup> | 0.802<sup>a</sup> | 0.448<sup>a</sup> | 0.957 |
| Laya noul: P(yes) | 0.476 | 0.438 | 0.614<sup>a</sup> | 0.679 | 0.297 | 0.682 |
| Laya score: E[y] | 0.433<sup>bc</sup> | 0.389<sup>bc</sup> | 0.592<sup>abc</sup> | 0.618<sup>bc</sup> | 0.263<sup>bcd</sup> | 0.634 |
| Laya score: E[2^y-1] | 0.429<sup>bc</sup> | 0.386<sup>bc</sup> | 0.587<sup>abc</sup> | 0.611<sup>bc</sup> | 0.257<sup>bcd</sup> | 0.632 |
| Laya score: P(y>=2) | 0.433<sup>bc</sup> | 0.389<sup>bc</sup> | 0.590<sup>abc</sup> | 0.585<sup>bc</sup> | 0.263<sup>bcd</sup> | 0.638 |
| Jev noul: P(yes) | 0.683<sup>ad</sup> | 0.637<sup>ad</sup> | 0.741<sup>ad</sup> | 0.838<sup>ad</sup> | 0.476<sup>ad</sup> | 0.857 |
| Jev noul, ties by E[y] | 0.674<sup>ad</sup> | 0.624<sup>ad</sup> | 0.738<sup>ad</sup> | 0.821<sup>ad</sup> | 0.474<sup>ad</sup> | 0.847 |
| Jev score: E[y] | 0.695<sup>ad</sup> | 0.650<sup>ad</sup> | 0.751<sup>acd</sup> | 0.851<sup>ad</sup> | 0.490<sup>ad</sup> | 0.868 |

## Laya score: ranking rule x temperature (nDCG@10)

|                    |   dl19 |   dl20 |
|:-------------------|-------:|-------:|
| ('exp', 'fitted')  | 0.4938 | 0.3909 |
| ('exp', 'raw')     | 0.4853 | 0.3835 |
| ('exp', 'shipped') | 0.487  | 0.3856 |
| ('ge2', 'fitted')  | 0.4797 | 0.3857 |
| ('ge2', 'raw')     | 0.4869 | 0.3898 |
| ('ge2', 'shipped') | 0.4962 | 0.3886 |
| ('lin', 'fitted')  | 0.4936 | 0.3921 |
| ('lin', 'raw')     | 0.4883 | 0.3893 |
| ('lin', 'shipped') | 0.4875 | 0.3977 |

## Ties among a query's pool scores (broken by BM25 rank)

|                                |   share of pairs tied |
|:-------------------------------|----------------------:|
| BM25 (bm25s order of the pool) |                0.5802 |
| monoT5-base                    |                0.0004 |
| Qwen3-Reranker-0.6B            |                0.0002 |
| Laya noul: P(yes)              |                0      |
| Laya score: E[y]               |                0      |
| Laya score: E[2^y-1]           |                0      |
| Laya score: P(y>=2)            |                0      |
| Jev noul: P(yes)               |                0.9803 |
| Jev noul, ties by E[y]         |                0.737  |
| Jev score: E[y]                |                0.9144 |

## Tie robustness (DL19+20): random tie-break, mean of 20 draws

| system                 |   share tied |   nDCG@10 (BM25 tie-break) |   nDCG@10 (random tie-break) |   nDCG'@10 (random tie-break) |
|:-----------------------|-------------:|---------------------------:|-----------------------------:|------------------------------:|
| Jev noul: P(yes)       |        0.98  |                     0.683  |                       0.6676 |                        0.7318 |
| Jev noul, ties by E[y] |        0.737 |                     0.6736 |                       0.6725 |                        0.7369 |
| Jev score: E[y]        |        0.914 |                     0.6953 |                       0.693  |                        0.7501 |

## Temperatures (keyed by the collection they are applied to)

```
{'raw': {'noul': {'dl19': 1.0, 'dl20': 1.0}, 'score': {'dl19': 1.0, 'dl20': 1.0}}, 'shipped': {'noul': {'dl19': 1.9834, 'dl20': 1.9834}, 'score': {'dl19': 1.2514, 'dl20': 1.2514}}, 'fitted': {'noul': {'dl19': 2.16, 'dl20': 2.8284}, 'score': {'dl19': 4.6646, 'dl20': 4.4641}}, 'monot5_xfit': {'dl19': 4.539, 'dl20': 6.0456}, 'qwen3_xfit': {'dl19': 5.505, 'dl20': 4.6721}, 'laya_noul_xfit': {'dl19': 2.162, 'dl20': 2.8312}, 'jev_noul_xfit': {'dl19': 1.3184, 'dl20': 1.3867}}
```
