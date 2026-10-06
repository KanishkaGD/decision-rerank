# Effectiveness (official TREC DL rerank pool, up to 1000 per query)

Superscripts: significantly different (paired t-test, Holm-corrected p < 0.05) from a = BM25, b = monoT5-base, c = Qwen3-Reranker-0.6B.

## DL19

| System | nDCG@10 | nDCG@10 (exp) | RR@10 | AP@1000 | Judged@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.429 | 0.000 | 0.662 | 0.274 | 0.991 |
| monoT5-base | 0.660<sup>a</sup> | 0.000 | 0.890<sup>a</sup> | 0.457<sup>a</sup> | 0.974 |
| Qwen3-Reranker-0.6B | 0.631<sup>a</sup> | 0.000 | 0.813 | 0.447<sup>a</sup> | 0.949 |
| Laya noul: P(yes) | 0.468<sup>bc</sup> | 0.000 | 0.743 | 0.323<sup>bc</sup> | 0.702 |
| Laya score: E[y] | 0.433<sup>bc</sup> | 0.000 | 0.667<sup>b</sup> | 0.295<sup>bc</sup> | 0.656 |
| Laya score: E[2^y-1] | 0.427<sup>bc</sup> | 0.000 | 0.662<sup>b</sup> | 0.291<sup>bc</sup> | 0.656 |
| Laya score: P(y>=2) | 0.437<sup>bc</sup> | 0.000 | 0.672<sup>b</sup> | 0.296<sup>bc</sup> | 0.653 |

## DL20

| System | nDCG@10 | nDCG@10 (exp) | RR@10 | AP@1000 | Judged@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.441 | 0.000 | 0.623 | 0.268 | 0.970 |
| monoT5-base | 0.678<sup>a</sup> | 0.000 | 0.866<sup>a</sup> | 0.478<sup>a</sup> | 0.996 |
| Qwen3-Reranker-0.6B | 0.632<sup>a</sup> | 0.000 | 0.806<sup>a</sup> | 0.452<sup>a</sup> | 0.965 |
| Laya noul: P(yes) | 0.412<sup>bc</sup> | 0.000 | 0.626<sup>bc</sup> | 0.280<sup>bc</sup> | 0.665 |
| Laya score: E[y] | 0.357<sup>bc</sup> | 0.000 | 0.577<sup>bc</sup> | 0.237<sup>bc</sup> | 0.624 |
| Laya score: E[2^y-1] | 0.348<sup>bc</sup> | 0.000 | 0.561<sup>bc</sup> | 0.230<sup>bc</sup> | 0.619 |
| Laya score: P(y>=2) | 0.354<sup>bc</sup> | 0.000 | 0.505<sup>bc</sup> | 0.238<sup>bc</sup> | 0.637 |

## DL19+20

| System | nDCG@10 | nDCG@10 (exp) | RR@10 | AP@1000 | Judged@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 (bm25s order of the pool) | 0.436 | 0.000 | 0.640 | 0.271 | 0.979 |
| monoT5-base | 0.670<sup>a</sup> | 0.000 | 0.877<sup>a</sup> | 0.469<sup>a</sup> | 0.987 |
| Qwen3-Reranker-0.6B | 0.632<sup>a</sup> | 0.000 | 0.809<sup>a</sup> | 0.450<sup>a</sup> | 0.958 |
| Laya noul: P(yes) | 0.437<sup>bc</sup> | 0.000 | 0.678<sup>bc</sup> | 0.299<sup>bc</sup> | 0.681 |
| Laya score: E[y] | 0.391<sup>bc</sup> | 0.000 | 0.617<sup>bc</sup> | 0.262<sup>bc</sup> | 0.638 |
| Laya score: E[2^y-1] | 0.383<sup>bc</sup> | 0.000 | 0.606<sup>bc</sup> | 0.257<sup>bc</sup> | 0.635 |
| Laya score: P(y>=2) | 0.391<sup>bc</sup> | 0.000 | 0.579<sup>bc</sup> | 0.263<sup>bc</sup> | 0.644 |

## Laya score: ranking rule x temperature (nDCG@10)

|                    |   dl19 |   dl20 |
|:-------------------|-------:|-------:|
| ('exp', 'fitted')  | 0.4327 | 0.3555 |
| ('exp', 'raw')     | 0.4272 | 0.3483 |
| ('exp', 'shipped') | 0.4307 | 0.3486 |
| ('ge2', 'fitted')  | 0.4225 | 0.3508 |
| ('ge2', 'raw')     | 0.4371 | 0.3536 |
| ('ge2', 'shipped') | 0.4384 | 0.3508 |
| ('lin', 'fitted')  | 0.4304 | 0.3579 |
| ('lin', 'raw')     | 0.4329 | 0.3573 |
| ('lin', 'shipped') | 0.432  | 0.355  |

## Ties among a query's pool scores (broken by BM25 rank)

|                                |   share of pairs tied |
|:-------------------------------|----------------------:|
| BM25 (bm25s order of the pool) |                0.5802 |
| monoT5-base                    |                0.0004 |
| Qwen3-Reranker-0.6B            |                0.8939 |
| Laya noul: P(yes)              |                0.3203 |
| Laya score: E[y]               |                0      |
| Laya score: E[2^y-1]           |                0      |
| Laya score: P(y>=2)            |                0      |

## Laya temperatures per condition

```
{'raw': {'noul': [1.0], 'score': [1.0]}, 'shipped': {'noul': [1.9834], 'score': [1.2514]}, 'fitted': {'noul': [2.1599, 2.8326], 'score': [4.4302, 4.6209]}}
```
