"""
Paths and experiment-wide settings. Paths come from the environment (CDM_DATA, CDM_RUNS, CDM_OUTPUTS, CDM_RESULTS);
the defaults are relative to the working directory.
"""

import os

CDM_ROOT = os.environ.get("CDM_ROOT", ".")
# Downloads that aren't Pyserini or Hugging Face caches, e.g. official rerank pools: data/pools/<dataset>/.
DATA_FOLDER = os.environ.get("CDM_DATA", "data")
POOLS_FOLDER = os.path.join(DATA_FOLDER, "pools")
# Cached first-stage candidates: runs/<dataset>/<first-stage>/ (src/cdm/runs.py).
RUNS_FOLDER = os.environ.get("CDM_RUNS", "runs")
# Raw reranker outputs: outputs/<dataset>/<first-stage>/<system>/<phrasing>/ -- every score, per-query metrics,
# the reranked run and a manifest. Large, gitignored.
OUTPUTS_FOLDER = os.environ.get("CDM_OUTPUTS", "outputs")
# results.csv, one row per evaluated configuration (src/cdm/results_csv.py). Tracked in git.
RESULTS_FOLDER = os.environ.get("CDM_RESULTS", "results")

SEED = 42

# First stage where a dataset has no official rerank pool (datasets.Dataset.pool): BM25 top-100 (research plan,
# "Experimental setup").
FIRST_STAGE_DEPTH = 100

# Effectiveness metrics. Binary metrics use the dataset's relevance threshold (datasets.Dataset.rel_threshold),
# so they are built per dataset by evaluation.effectiveness_measures(); nDCG uses the graded labels.
NDCG_CUTOFFS = [10]
RECALL_CUTOFFS = [10, 100]

# Calibration: equal-width bins for the expected calibration error (ECE).
ECE_BINS = 15
