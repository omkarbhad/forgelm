"""Generate text from a trained checkpoint.

    python -m forgelm.generate --run runs/tiny --prompt "ROMEO:"
"""

import argparse
import os

import torch

from .config import ForgeConfig
from .model import ForgeLM
from .tokenizer import BPETokenizer
from .train import pick_device


def load_run(run_dir: str, device: str = "cpu"):
    ckpt = torch.load(os.path.join(run_dir, "ckpt.pt"), map_location=device)
    model = ForgeLM(ForgeConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, BPETokenizer.load(os.path.join(run_dir, "tokenizer.json"))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/tiny")
    p.add_argument("--prompt", default="\n")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto")
    a = p.parse_args(argv)

    if a.seed is not None:
        torch.manual_seed(a.seed)
    device = pick_device(a.device)
    model, tok = load_run(a.run, device)
    idx = torch.tensor([tok.encode(a.prompt)], device=device)
    out = model.generate(idx, a.max_new_tokens, a.temperature, a.top_k, a.top_p, tok.eot_id)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
