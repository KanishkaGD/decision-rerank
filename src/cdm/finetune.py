"""
Our own reranker fine-tune (docs/notes/reranker-27b-plan.md): LoRA on a Qwen backbone, trained to answer Qwen3-Reranker's
prompt with "yes" for relevant and "no" for non-relevant passages, so the existing reader (Qwen3Reranker with
`adapter`) scores the result like Qwen3-Reranker and Prism: z = z_yes - z_no at the first answer token.

One training step sees `groups_per_step` query groups (1 positive + k hard negatives each, src/cdm/finetune_data.py),
spread over the data-parallel ranks and `grad_accum` micro-steps. Losses on the group's log-odds z:
  listwise   cross-entropy of the positive among its group (InfoNCE, temperature 1) -- RankLLaMA's objective and
             SWIFT's listwise generative_reranker loss;
  pointwise  binary cross-entropy of every passage's z against its label (Qwen3-Reranker's documented pointwise loss);
  both       their sum.

Data parallel without DDP's wrapper: every rank holds the frozen base model and the LoRA weights, and the LoRA
gradients are averaged with one all-reduce per optimizer step (they are a few hundred MB even at 27B).

Checkpoints: <out>/checkpoint-<step>/ holds the adapter (kept, for selection on msmarco-dev) and trainer_state.pt
(optimizer, scheduler, RNG, step; only the newest is kept). --resume continues from the newest checkpoint with the same
data order; a job stops by itself, with a checkpoint, before its Slurm time limit (SLURM_JOB_END_TIME), so a chain of
jobs (`slurm/submit.sh finetune ... --after <id>`) finishes a run longer than 48 h. <out>/final/ is written at the end.
"""

import glob
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Every linear layer of the language model -- attention (q/k/v/o), Gated DeltaNet (in_proj_qkv/z, out_proj) and MLP
# -- and none of the vision tower or the multi-token-prediction head of the Qwen3.5/3.8 releases. A regex
# (PEFT full-matches it against module names).
LORA_TARGETS = (
    r"^(?!.*(visual|mtp)).*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$"
)


@dataclass
class TrainConfig:
    system: str
    model_id: str
    revision: Optional[str]
    data: str  # groups.jsonl.gz
    out: str
    instruction: str
    system_prompt: str
    loss: str = "listwise"
    group_size: int = 16  # 1 positive + group_size - 1 negatives (the first ones of each stored group)
    max_length: int = 1024  # the reader's max_length: passages are cut exactly as at scoring time
    lr: float = 1e-4
    warmup_ratio: float = 0.03
    scheduler: str = "linear"
    weight_decay: float = 0.0
    epochs: float = 1.0
    groups_per_step: int = 64  # effective batch in query groups (across ranks and micro-steps)
    micro_groups: int = 1  # query groups per forward pass on one rank
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    dtype: str = "bfloat16"  # frozen base weights; LoRA weights in fp32, forward under bf16 autocast
    gradient_checkpointing: bool = True
    max_groups: Optional[int] = None  # only the first N groups of the (shuffled) data: smoke tests, learning curves
    max_steps: Optional[int] = None  # stop after N optimizer steps (throughput probes)
    save_steps: int = 200
    log_steps: int = 10
    seed: int = 42
    max_hours: Optional[float] = None
    stop_margin_min: float = 20.0  # stop this long before the Slurm time limit
    write_final: bool = True


# ------------------------------------------------------------------------------------------------- pure helpers


def group_passages(group: Dict, group_size: int) -> Tuple[List[str], List[float]]:
    """Texts and labels of one stored group, cut to group_size: the positive first, then the first negatives."""
    negs = group["negs"][: group_size - 1]
    if len(negs) < group_size - 1:
        raise ValueError(f"group {group['qid']} has {len(negs)} negatives, need {group_size - 1}")
    return [group["pos"]["text"]] + [n["text"] for n in negs], [1.0] + [0.0] * len(negs)


def steps_for(n_groups: int, groups_per_step: int, epochs: float) -> int:
    return max(1, math.floor(n_groups * epochs / groups_per_step))


def data_order(n_groups: int, epochs: float, seed: int) -> List[int]:
    """Group indices in training order: one seeded permutation per (started) epoch, concatenated."""
    order: List[int] = []
    for epoch in range(math.ceil(epochs)):
        perm = list(range(n_groups))
        random.Random(seed + epoch).shuffle(perm)
        order.extend(perm)
    return order


def step_groups(order: Sequence[int], step: int, groups_per_step: int, rank: int, world: int) -> List[int]:
    """The groups rank `rank` trains on in optimizer step `step` (0-based)."""
    batch = order[step * groups_per_step : (step + 1) * groups_per_step]
    per_rank = groups_per_step // world
    return list(batch[rank * per_rank : (rank + 1) * per_rank])


def group_loss(z, labels, group_size: int, loss: str):
    """Mean loss over the groups in z (shape [n_groups * group_size], positives at index 0 of each group)."""
    import torch
    import torch.nn.functional as F

    zg = z.view(-1, group_size)
    total = z.new_zeros(())
    if loss in ("listwise", "both"):
        total = total + F.cross_entropy(zg, torch.zeros(zg.shape[0], dtype=torch.long, device=z.device))
    if loss in ("pointwise", "both"):
        total = total + F.binary_cross_entropy_with_logits(z, labels)
    if loss not in ("listwise", "pointwise", "both"):
        raise ValueError(f"loss must be listwise, pointwise or both, not '{loss}'")
    return total


def deadline(max_hours: Optional[float], margin_min: float, start: float) -> Optional[float]:
    """
    When to stop: the earlier of start + max_hours and the Slurm job's end minus the margin (None: never). The margin
    is at most a tenth of the job's length, so short jobs (smoke tests) still train.
    """
    ends = []
    if max_hours:
        ends.append(start + max_hours * 3600)
    end = os.environ.get("SLURM_JOB_END_TIME", "")
    if end.isdigit():
        begin = os.environ.get("SLURM_JOB_START_TIME", "")
        margin = margin_min * 60
        if begin.isdigit():
            margin = min(margin, 0.1 * (int(end) - int(begin)))
        ends.append(int(end) - margin)
    return min(ends) if ends else None


def latest_checkpoint(out: str) -> Optional[str]:
    """The newest <out>/checkpoint-<step> with a trainer_state.pt (one that can be resumed)."""
    ckpts = [
        p for p in glob.glob(os.path.join(out, "checkpoint-*")) if os.path.exists(os.path.join(p, "trainer_state.pt"))
    ]
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[1])) if ckpts else None


# ------------------------------------------------------------------------------------------------------ training


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def train(cfg: TrainConfig, resume: bool = False) -> Dict:
    import numpy as np
    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from transformers import AutoTokenizer, get_scheduler

    from cdm.finetune_data import read_groups
    from cdm.provenance import write_manifest
    from cdm.rerankers.pointwise_lm import QWEN3_SUFFIX, load_causal_lm, qwen3_prefix, qwen3_reranker_ids

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")  # gloo: CPU tests of this path
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if cfg.groups_per_step % (world * cfg.micro_groups):
        raise ValueError(
            f"groups_per_step {cfg.groups_per_step} must be a multiple of ranks x micro_groups ({world} x "
            f"{cfg.micro_groups})"
        )
    grad_accum = cfg.groups_per_step // (world * cfg.micro_groups)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    start = time.time()
    stop_at = deadline(cfg.max_hours, cfg.stop_margin_min, start)

    groups = read_groups(cfg.data)
    order = data_order(len(groups), cfg.epochs, cfg.seed)
    if cfg.max_groups:
        order = order[: cfg.max_groups]
        n_train = cfg.max_groups
    else:
        n_train = len(groups)
    total_steps = steps_for(n_train, cfg.groups_per_step, cfg.epochs if not cfg.max_groups else 1.0)
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, revision=cfg.revision, padding_side="left")
    prefix = tokenizer.encode(qwen3_prefix(cfg.system_prompt), add_special_tokens=False)
    suffix = tokenizer.encode(QWEN3_SUFFIX, add_special_tokens=False)
    yes_id, no_id = tokenizer.convert_tokens_to_ids("yes"), tokenizer.convert_tokens_to_ids("no")

    # Straight onto this rank's GPU: eight ranks each staging 55 GB of 27B weights in host memory would not fit.
    model = load_causal_lm(
        cfg.model_id, cfg.revision, cfg.dtype, **({"device_map": {"": device}} if device.type == "cuda" else {})
    )
    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora)
    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    _log(rank, f"LoRA: {n_trainable / 1e6:.1f}M trainable parameters; {world} rank(s), grad_accum {grad_accum}")

    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=device.type == "cuda")
    scheduler = get_scheduler(
        cfg.scheduler,
        optimizer,
        num_warmup_steps=round(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    step = 0
    ckpt = latest_checkpoint(cfg.out) if resume else None
    if ckpt:
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(os.path.join(ckpt, "adapter_model.safetensors")))
        state = torch.load(os.path.join(ckpt, "trainer_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(state["cuda_rng"][min(rank, len(state["cuda_rng"]) - 1)])
        _log(rank, f"Resumed from {ckpt} at step {step}/{total_steps}")
    elif resume:
        _log(rank, f"--resume: no checkpoint in {cfg.out}, starting at step 0")
    if step >= total_steps:
        _log(rank, f"Already trained ({step}/{total_steps} steps); nothing to do.")
        if cfg.write_final and rank == 0 and not os.path.exists(os.path.join(cfg.out, "final", "adapter_config.json")):
            _save_adapter(model, os.path.join(cfg.out, "final"), cfg)
        if world > 1:
            dist.barrier()
            dist.destroy_process_group()
        return {"step": step, "total_steps": total_steps, "finished": True}

    if rank == 0:
        # The run's manifest; a resumed job writes its own next to it (resume-<step>/manifest.json).
        write_manifest(
            os.path.join(cfg.out, f"resume-{step}") if ckpt else cfg.out,
            job="finetune",
            **asdict(cfg),
            resume_from=ckpt,
            world_size=world,
            grad_accum=grad_accum,
            total_steps=total_steps,
            n_groups=len(groups),
            n_trainable=n_trainable,
        )

    def encode(batch_groups: List[Dict]):
        ids, labels = [], []
        for g in batch_groups:
            texts, lab = group_passages(g, cfg.group_size)
            ids += qwen3_reranker_ids(tokenizer, prefix, suffix, cfg.instruction, g["query"], texts, cfg.max_length)
            labels += lab
        enc = tokenizer.pad({"input_ids": ids}, return_tensors="pt")
        n_tokens = sum(len(x) for x in ids)
        return enc, torch.tensor(labels), n_tokens, enc["input_ids"].numel()

    model.train()
    log_path = os.path.join(cfg.out, "train_log.jsonl")
    t_log, tokens_log, padded_log, losses = time.time(), 0, 0, []
    stopped_early = False
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    while step < total_steps:
        mine = step_groups(order, step, cfg.groups_per_step, rank, world)
        step_loss = 0.0
        for m in range(grad_accum):
            batch = [groups[i] for i in mine[m * cfg.micro_groups : (m + 1) * cfg.micro_groups]]
            enc, labels, n_tok, n_pad = encode(batch)
            enc, labels = enc.to(device), labels.to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
            z = logits[:, yes_id].float() - logits[:, no_id].float()
            loss = group_loss(z, labels, cfg.group_size, cfg.loss) / grad_accum
            loss.backward()
            step_loss += loss.item()
            tokens_log += n_tok
            padded_log += n_pad
        if world > 1:
            # SUM / world rather than ReduceOp.AVG, which gloo lacks.
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad)
                    p.grad /= world
            t = torch.tensor([step_loss], device=device)
            dist.all_reduce(t)
            step_loss = t.item() / world
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        losses.append(step_loss)

        if step % cfg.log_steps == 0 or step == total_steps:
            dt = time.time() - t_log
            rec = {
                "step": step,
                "total_steps": total_steps,
                "loss": sum(losses) / len(losses),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm,
                "s_per_step": dt / len(losses),
                # Real (unpadded) tokens per second over all ranks; padded tokens are what the GPU processed.
                "tokens_per_s": tokens_log * world / dt,
                "padded_tokens_per_s": padded_log * world / dt,
                "max_mem_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0,
                "elapsed_h": (time.time() - start) / 3600,
            }
            _log(rank, json.dumps(rec))
            if rank == 0:
                with open(log_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            t_log, tokens_log, padded_log, losses = time.time(), 0, 0, []

        stop_now = stop_at is not None and time.time() > stop_at
        if world > 1:
            flag = torch.tensor([1.0 if stop_now else 0.0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop_now = flag.item() > 0
        if (cfg.save_steps and step % cfg.save_steps == 0) or (stop_now and step < total_steps):
            if rank == 0:
                _save_checkpoint(model, optimizer, scheduler, step, cfg, world)
            if world > 1:
                dist.barrier()
        if stop_now and step < total_steps:
            _log(rank, f"Stopping at step {step}/{total_steps} before the time limit; resume with --resume.")
            stopped_early = True
            break

    if not stopped_early and rank == 0:
        if cfg.save_steps and step % cfg.save_steps != 0:
            _save_checkpoint(model, optimizer, scheduler, step, cfg, world)
        if cfg.write_final:
            _save_adapter(model, os.path.join(cfg.out, "final"), cfg)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return {"step": step, "total_steps": total_steps, "finished": not stopped_early}


def _save_adapter(model, path: str, cfg: TrainConfig) -> None:
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    manifest = os.path.join(cfg.out, "manifest.json")
    if os.path.exists(manifest):
        shutil.copyfile(manifest, os.path.join(path, "manifest.json"))


def _save_checkpoint(model, optimizer, scheduler, step: int, cfg: TrainConfig, world: int) -> None:
    """Adapter + trainer state; older checkpoints keep their adapter but lose their (large) trainer state."""
    import torch

    path = os.path.join(cfg.out, f"checkpoint-{step}")
    _save_adapter(model, path, cfg)
    cuda_rng = None
    if torch.cuda.is_available():
        cuda_rng = [torch.cuda.get_rng_state()] * world
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": cuda_rng,
        },
        os.path.join(path, "trainer_state.pt.tmp"),
    )
    os.replace(os.path.join(path, "trainer_state.pt.tmp"), os.path.join(path, "trainer_state.pt"))
    for old in glob.glob(os.path.join(cfg.out, "checkpoint-*", "trainer_state.pt")):
        if os.path.dirname(old) != path:
            os.remove(old)
    print(f"Saved {path}", flush=True)
