# Query performance prediction from relevance probabilities

Kendall tau / Pearson r with the true per-query metric; per collection and their mean. MAE: |expected P@10 - actual P@10| (probability predictors only). 'Labels' marks predictors that need judged pairs (cross-fitted on the other collection).

## BM25 top 10 (mean actual nDCG@10 0.488, P@10 0.365)

| Predictor | Labels | tau nDCG@10 DL19 | DL20 | mean | r nDCG@10 mean | tau P@10 mean | MAE P@10 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev E[grade] |  | 0.426 | 0.529 | 0.478 | 0.682 | 0.541 |  |
| Jev P(yes), Platt | yes | 0.464 | 0.461 | 0.462 | 0.684 | 0.536 | 0.162 |
| Jev P(yes) |  | 0.409 | 0.469 | 0.439 | 0.662 | 0.526 | 0.158 |
| monoT5 mean log-odds@10 |  | 0.411 | 0.452 | 0.432 | 0.655 | 0.440 |  |
| monoT5 P(true), Platt | yes | 0.386 | 0.454 | 0.420 | 0.612 | 0.447 | 0.205 |
| monoT5 P(true), raw |  | 0.393 | 0.430 | 0.412 | 0.603 | 0.455 | 0.244 |
| Qwen3-0.6B P(yes), Platt | yes | 0.369 | 0.413 | 0.391 | 0.580 | 0.395 | 0.213 |
| Qwen3-0.6B P(yes), raw |  | 0.318 | 0.350 | 0.334 | 0.530 | 0.329 | 0.464 |
| Laya P(yes) (shipped T) |  | 0.355 | 0.313 | 0.334 | 0.505 | 0.425 | 0.183 |
| Laya E[grade] (shipped T) |  | 0.320 | 0.289 | 0.304 | 0.476 | 0.378 |  |
| BM25 sigma@100 |  | 0.231 | 0.376 | 0.303 | 0.427 | 0.267 |  |
| Kev-0.8B E[grade] |  | 0.256 | 0.245 | 0.251 | 0.332 | 0.257 |  |
| BM25 NQC-style (sigma@100 / pool mean) |  | 0.169 | 0.294 | 0.232 | 0.411 | 0.186 |  |
| Kev-0.8B P(yes) (shipped T) |  | 0.227 | 0.188 | 0.208 | 0.266 | 0.187 | 0.315 |
| BM25 top score |  | 0.085 | 0.169 | 0.127 | 0.232 | 0.109 |  |

## monoT5 top 10 (mean actual nDCG@10 0.718, P@10 0.585)

| Predictor | Labels | tau nDCG@10 DL19 | DL20 | mean | r nDCG@10 mean | tau P@10 mean | MAE P@10 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev E[grade] |  | 0.292 | 0.309 | 0.300 | 0.354 | 0.454 |  |
| Jev P(yes), Platt | yes | 0.294 | 0.278 | 0.286 | 0.381 | 0.425 | 0.210 |
| Jev P(yes) |  | 0.241 | 0.293 | 0.267 | 0.325 | 0.421 | 0.200 |
| monoT5 mean log-odds@10 |  | 0.146 | 0.278 | 0.212 | 0.250 | 0.330 |  |
| monoT5 P(true), Platt | yes | 0.141 | 0.281 | 0.211 | 0.208 | 0.335 | 0.266 |
| monoT5 P(true), raw |  | 0.141 | 0.260 | 0.200 | 0.014 | 0.357 | 0.360 |
| Qwen3-0.6B P(yes), Platt | yes | 0.148 | 0.227 | 0.188 | 0.257 | 0.287 | 0.264 |
| Qwen3-0.6B P(yes), raw |  | 0.174 | 0.188 | 0.181 | 0.014 | 0.289 | 0.386 |
| Laya P(yes) (shipped T) |  | 0.128 | 0.191 | 0.159 | 0.262 | 0.305 | 0.224 |
| Laya E[grade] (shipped T) |  | 0.134 | 0.171 | 0.153 | 0.268 | 0.315 |  |
| Kev-0.8B E[grade] |  | 0.012 | 0.073 | 0.043 | 0.091 | 0.097 |  |
| Kev-0.8B P(yes) (shipped T) |  | 0.059 | 0.012 | 0.035 | 0.134 | 0.156 | 0.250 |
