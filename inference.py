import argparse

import torch

from model import GPTConfig, MiniGPT


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="minigpt.pt")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--gen-len", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()

def decode(token_ids, itos):
    return "".join([itos[i] for i in token_ids])


def load_config(checkpoint, stoi):
    if "config" in checkpoint:
        return GPTConfig(**checkpoint["config"])

    return GPTConfig(
        vocab_size=len(stoi),
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
    )


def resolve_prompt(args):
    if args.prompt is not None:
        return args.prompt

    user_prompt = input("prompt> ")
    if user_prompt.strip():
        return user_prompt
    return "MiniGPT "


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location=device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    config = load_config(checkpoint, stoi)

    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt = resolve_prompt(args)
    prompt_ids = [stoi[ch] for ch in prompt if ch in stoi]
    if len(prompt_ids) == 0:
        prompt_ids = [0]

    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        start,
        max_new_tokens=args.gen_len,
        temperature=args.temperature,
        top_k=args.top_k,
    )[0].tolist()

    print("===== inference output =====")
    print(decode(output_ids, itos))


if __name__ == "__main__":
    main()
