"""The two Laya questions, fixed before any run and never tuned on DL queries.

The state is ``{"query": ..., "passage": ...}`` (Laya serialises a dict state as JSON), so the
instructions name its fields. The score rubric is UMBRELA's four levels (Upadhyay et al. 2024),
which mirror the TREC DL passage grades 0-3.
"""

NOUL = {"type": "noul", "instructions": "Does the 'passage' answer the 'query'?"}

UMBRELA = [
    "The passage has nothing to do with the query.",
    "The passage seems related to the query but does not answer it.",
    "The passage has some answer for the query, but the answer may be a bit unclear, "
    "or hidden amongst extraneous information.",
    "The passage is dedicated to the query and contains the exact answer.",
]
SCORE = {"type": "score", "instructions": "How well does the 'passage' answer the 'query'?",
         "criteria": UMBRELA}

QUESTIONS = {"noul": NOUL, "score": SCORE}


def state(query: str, passage: str) -> dict:
    return {"query": query, "passage": passage}
