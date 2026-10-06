# Query performance prediction from relevance probabilities

Kendall tau / Pearson r with the true per-query metric; per collection and their mean. MAE: |expected P@10 - actual P@10| (probability predictors only). 'Labels' marks predictors that need judged pairs (cross-fitted on the other collection).

## BM25 standard run top 10 (mean actual nDCG@10 0.491, P@10 0.377)

| Predictor | Labels | tau nDCG@10 DL19 | DL20 | mean | r nDCG@10 mean | tau P@10 mean | MAE P@10 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev E[grade] |  | 0.373 | 0.555 | 0.464 | 0.695 | 0.487 |  |
| Qwen3.7-Flash E[grade] |  | 0.395 | 0.520 | 0.457 | 0.670 | 0.484 |  |
| Jev P(yes), Platt | yes | 0.371 | 0.539 | 0.455 | 0.673 | 0.516 | 0.162 |
| Jev P(yes) |  | 0.353 | 0.544 | 0.449 | 0.647 | 0.506 | 0.165 |
| Qwen3.7-Flash P(yes) |  | 0.391 | 0.497 | 0.444 | 0.630 | 0.509 | 0.170 |
| monoT5 P(true), Platt | yes | 0.353 | 0.514 | 0.434 | 0.614 | 0.447 | 0.203 |
| monoT5 mean log-odds@10 |  | 0.360 | 0.507 | 0.433 | 0.605 | 0.454 |  |
| Qwen3-0.6B P(yes), Platt | yes | 0.373 | 0.468 | 0.421 | 0.622 | 0.405 | 0.212 |
| monoT5 P(true), raw |  | 0.338 | 0.475 | 0.406 | 0.581 | 0.438 | 0.236 |
| Qwen3-0.6B P(yes), raw |  | 0.347 | 0.420 | 0.383 | 0.566 | 0.364 | 0.440 |
| BM25 sigma@100 |  | 0.322 | 0.412 | 0.367 | 0.494 | 0.294 |  |
| Laya E[grade] (shipped T) |  | 0.322 | 0.390 | 0.356 | 0.523 | 0.371 |  |
| BM25 NQC-style (sigma@100 / mean@1000) |  | 0.293 | 0.401 | 0.347 | 0.525 | 0.263 |  |
| Laya P(yes) (shipped T) |  | 0.305 | 0.378 | 0.341 | 0.490 | 0.408 | 0.175 |
| Kev-0.8B E[grade] |  | 0.274 | 0.272 | 0.273 | 0.419 | 0.236 |  |
| Kev-0.8B P(yes) (shipped T) |  | 0.262 | 0.205 | 0.234 | 0.348 | 0.230 | 0.308 |
| BM25 top score |  | 0.192 | 0.134 | 0.163 | 0.286 | 0.133 |  |

## BM25 (our pool order) top 10 (mean actual nDCG@10 0.488, P@10 0.365)

| Predictor | Labels | tau nDCG@10 DL19 | DL20 | mean | r nDCG@10 mean | tau P@10 mean | MAE P@10 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev E[grade] |  | 0.426 | 0.529 | 0.478 | 0.682 | 0.541 |  |
| Jev P(yes), Platt | yes | 0.464 | 0.461 | 0.462 | 0.684 | 0.536 | 0.162 |
| Qwen3.7-Flash E[grade] |  | 0.424 | 0.469 | 0.447 | 0.650 | 0.511 |  |
| Jev P(yes) |  | 0.409 | 0.469 | 0.439 | 0.662 | 0.526 | 0.158 |
| Qwen3.7-Flash P(yes) |  | 0.433 | 0.431 | 0.432 | 0.618 | 0.492 | 0.168 |
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
| Qwen3.7-Flash E[grade] |  | 0.314 | 0.358 | 0.336 | 0.378 | 0.464 |  |
| Qwen3.7-Flash P(yes) |  | 0.277 | 0.331 | 0.304 | 0.367 | 0.458 | 0.200 |
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
