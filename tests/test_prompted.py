"""The zero-shot base-model judge of the triplets: prompt assembly, truncation and the yes/no variants (toy tokenizer)."""

from cdm.rerankers.prompted import COMPLETION, DOC, QUESTION, PromptedJudge


class ToyTokenizer:
    """One token per character; 'yes', 'no' and their variants are single tokens."""

    words = {"yes": 1001, "Yes": 1002, " yes": 1003, "no": 2001, " no": 2003}

    def encode(self, text, add_special_tokens=False):
        return [self.words[text]] if text in self.words else [ord(c) for c in text]

    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None):
        ids = [self.encode(t) for t in texts]
        return {"input_ids": [x[:max_length] if truncation else x for x in ids]}


def judge(mode="completion", max_length=200):
    j = PromptedJudge.__new__(PromptedJudge)
    j.mode, j.max_length, j.tokenizer = mode, max_length, ToyTokenizer()
    j.prefix_text, j.suffix_text = j._template().split(DOC)
    return j


def test_question_is_s1ranks_without_field_references():
    assert "documents.D" not in QUESTION and "`" not in QUESTION and "directly addresses it" in QUESTION


def test_variants():
    j = judge()
    assert j._variants("yes") == [1001, 1002, 1003] and j._variants("no") == [2001, 2003]


def test_passage_cut_to_budget_prompt_kept():
    j = judge(max_length=400)
    (ids,) = j._ids("q?", ["x" * 500])
    assert len(ids) == 400
    text = "".join(chr(i) for i in ids)
    assert text.startswith("Query: q?\nDocument: x") and text.endswith("Answer:")
    assert COMPLETION.endswith("Answer:")
    assert j.count_truncated("q?", ["x" * 500, "short"]) == 1
