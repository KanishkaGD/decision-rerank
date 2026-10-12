"""The reranker fine-tuning trainer's pure parts: data order and sharding, losses, deadline, the shared reader input."""

import re

import pytest

from cdm.finetune import (
    LORA_TARGETS,
    data_order,
    deadline,
    group_passages,
    latest_checkpoint,
    step_groups,
    steps_for,
)
from cdm.rerankers import SELECTED_CHECKPOINT, SYSTEMS
from cdm.rerankers.pointwise_lm import QWEN3_INSTRUCTION, qwen3_reranker_ids


def test_lora_targets_language_model_only():
    hit = re.compile(LORA_TARGETS).fullmatch
    for name in [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.mlp.down_proj",
        "model.layers.5.self_attn.o_proj",  # text-only Qwen3
    ]:
        assert hit(name), name
    for name in [
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.0.mlp.up_proj",
        "mtp.layers.0.mlp.gate_proj",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "lm_head",
    ]:
        assert not hit(name), name


def test_group_passages():
    g = {"qid": "1", "pos": {"text": "P"}, "negs": [{"text": f"N{i}"} for i in range(5)]}
    assert group_passages(g, 4) == (["P", "N0", "N1", "N2"], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        group_passages(g, 8)


def test_order_steps_and_sharding():
    order = data_order(10, 1.0, seed=3)
    assert sorted(order) == list(range(10)) and order == data_order(10, 1.0, seed=3)
    assert len(data_order(10, 1.5, seed=3)) == 20
    assert steps_for(100, 8, 1.0) == 12 and steps_for(3, 8, 1.0) == 1
    order = list(range(16))
    shards = [step_groups(order, 1, 8, r, 2) for r in range(2)]
    assert shards == [[8, 9, 10, 11], [12, 13, 14, 15]]


def test_group_loss():
    torch = pytest.importorskip("torch")
    from cdm.finetune import group_loss

    z = torch.tensor([5.0, -5.0, -5.0, 0.0, 0.0, 0.0])
    labels = torch.tensor([1.0, 0, 0, 1, 0, 0])
    listwise = group_loss(z, labels, 3, "listwise")
    # group 1: positive far ahead (~0); group 2: uniform (log 3)
    assert abs(listwise.item() - (torch.log(torch.tensor(3.0)).item() + 0.0) / 2) < 1e-3
    pointwise = group_loss(z, labels, 3, "pointwise")
    both = group_loss(z, labels, 3, "both")
    assert abs(both.item() - listwise.item() - pointwise.item()) < 1e-6
    with pytest.raises(ValueError):
        group_loss(z, labels, 3, "nope")


def test_deadline(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_END_TIME", raising=False)
    monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)  # set when the tests run inside a Slurm job
    assert deadline(None, 20, 1000.0) is None
    assert deadline(1.0, 20, 1000.0) == 4600.0
    monkeypatch.setenv("SLURM_JOB_END_TIME", "4000")
    assert deadline(None, 20, 1000.0) == 2800.0
    assert deadline(1.0, 20, 1000.0) == 2800.0
    monkeypatch.setenv("SLURM_JOB_START_TIME", "1000")  # a 50-min job: margin capped at 5 min
    assert deadline(None, 20, 1000.0) == 3700.0


def test_latest_checkpoint(tmp_path):
    assert latest_checkpoint(str(tmp_path)) is None
    for step, state in [(200, False), (100, True), (50, True)]:
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if state:
            (d / "trainer_state.pt").write_bytes(b"x")
    assert latest_checkpoint(str(tmp_path)).endswith("checkpoint-100")


class ToyTokenizer:
    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None):
        ids = [[ord(c) for c in t] for t in texts]
        return {"input_ids": [x[:max_length] if truncation else x for x in ids]}


def test_reader_input_shared_with_trainer():
    ids = qwen3_reranker_ids(ToyTokenizer(), [1, 2], [3], QWEN3_INSTRUCTION, "q", ["doc", "x" * 500], 200)
    assert ids[0][:2] == [1, 2] and ids[0][-1] == 3
    body = "".join(chr(i) for i in ids[0][2:-1])
    assert body == f"<Instruct>: {QWEN3_INSTRUCTION}\n<Query>: q\n<Document>: doc"
    assert len(ids[1]) == 200  # cut to max_length, prefix and suffix kept


def test_own_rerankers_registered():
    for name in ["qwen3.5-9b-reranker"]:
        spec = SYSTEMS[name]
        assert spec.kind == "qwen3-reranker" and spec.revision
        # the checkpoint chosen on msmarco-dev where one was chosen (SELECTED_CHECKPOINT), else the final one
        assert spec.params["adapter"] == f"checkpoints/{name}/{SELECTED_CHECKPOINT.get(name, 'final')}"
    assert SELECTED_CHECKPOINT["qwen3.5-9b-reranker"] == "checkpoint-800"
