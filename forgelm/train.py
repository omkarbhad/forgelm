"""Train ForgeLM on a tokenized corpus.

    python -m forgelm.data                       # once: download + tokenize
    python -m forgelm.train --preset tiny        # trains, saves runs/<name>/ckpt.pt
"""

import argparse
import math
import os
import time

import torch

from . import data as data_lib
from .config import ForgeConfig, PRESETS
from .model import ForgeLM


def pick_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def lr_at(step: int, max_steps: int, lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay down to lr * min_ratio."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizer(model: ForgeLM, lr: float, weight_decay: float):
    # decay matrices (linear + embedding weights); leave norm gains alone
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model, data, batch_size, block_size, device, iters=20) -> float:
    model.eval()
    losses = [model(*data_lib.get_batch(data, batch_size, block_size, device))[1].item() for _ in range(iters)]
    model.train()
    return sum(losses) / len(losses)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="tiny", choices=PRESETS)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs/tiny")
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    device = pick_device(a.device)
    tok, train_data, val_data = data_lib.load(a.data_dir)
    cfg = ForgeConfig(vocab_size=tok.vocab_size, **PRESETS[a.preset])
    model = ForgeLM(cfg).to(device)
    opt = make_optimizer(model, a.lr, a.weight_decay)
    use_amp = device == "cuda"  # bf16 autocast on CUDA; fp32 elsewhere
    print(f"device {device} | {model.num_params() / 1e6:.2f}M params | vocab {cfg.vocab_size} | {a.max_steps} steps")

    os.makedirs(a.out_dir, exist_ok=True)
    tok.save(os.path.join(a.out_dir, "tokenizer.json"))
    best, t0 = float("inf"), time.time()
    for step in range(a.max_steps + 1):
        if step % a.eval_interval == 0 or step == a.max_steps:
            val = estimate_loss(model, val_data, a.batch_size, cfg.max_seq_len, device)
            print(f"step {step:5d} | val loss {val:.4f} | {time.time() - t0:6.0f}s")
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "step": step, "val_loss": val},
                           os.path.join(a.out_dir, "ckpt.pt"))
        if step == a.max_steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.max_steps, a.lr, a.warmup)
        x, y = data_lib.get_batch(train_data, a.batch_size, cfg.max_seq_len, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
        opt.step()
        if step % 50 == 0:
            print(f"step {step:5d} | train loss {loss.item():.4f}")
    print(f"best val loss {best:.4f}  ->  {os.path.join(a.out_dir, 'ckpt.pt')}")


if __name__ == "__main__":
    main()
