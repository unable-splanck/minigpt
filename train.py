import argparse
from pathlib import Path

import torch

from model import GPTConfig, MiniGPT

"""
# 使用方式
python train.py --data data.txt
"""

DEFAULT_CORPUS = (
    "miniGPT is a tiny transformer demo.\n"
    "we build it step by step.\n"
) * 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="minigpt.pt")
    parser.add_argument("--prompt", type=str, default="gpt ")
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def load_text(data_path):
    if data_path is None:
        return DEFAULT_CORPUS

    path = Path(data_path)
    if not path.exists():
        print(f"[warning] {path} 不存在，使用内置语料")
        return DEFAULT_CORPUS

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        print(f"[warning] {path} 为空，使用内置语料")
        return DEFAULT_CORPUS
    return text


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(token_ids, itos):
    return "".join([itos[i] for i in token_ids])


def get_batch(data, block_size, batch_size, device):
    if len(data) <= block_size + 1:
        raise ValueError("dataset is too short for the chosen block_size")

    idx = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in idx])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
    return x.to(device), y.to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = load_text(args.data)
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("==== MiniGPT Train ====")
    print("device:", device)
    print("data length:", len(data))
    print("vocab size:", len(stoi))
    print(
        "config:",
        {
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "block_size": args.block_size,
            "dropout": args.dropout,
        },
    )

    model.train()
    for step in range(1, args.max_steps + 1):
        xb, yb = get_batch(data, args.block_size, args.batch_size, device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    xb, yb = get_batch(data, args.block_size, args.batch_size, device)
    _, final_loss = model(xb, yb)
    print("final loss:", float(final_loss.detach()))

    ckpt = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "config": config.__dict__,
    }
    torch.save(ckpt, args.out)
    print("checkpoint saved to:", args.out)

    model.eval()
    prompt_ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if len(prompt_ids) == 0:
        prompt_ids = [0]
    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out_ids = model.generate(
        start,
        max_new_tokens=args.gen_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()
    out_text = decode(out_ids, itos)

    print("\n===== generated text =====")
    print(out_text)


if __name__ == "__main__":
    main()
