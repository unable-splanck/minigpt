import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="minigpt.pt")
    parser.add_argument("--prompt", type=str, default="MiniGPT ")
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    return parser.parse_args()


def decode(token_ids, itos):
    return "".join([itos[i] for i in token_ids])


class TinyLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("sequence length > block_size")

        tok = self.token_emb(idx)
        pos_ids = torch.arange(seq_len, device=idx.device)
        pos = self.pos_emb(pos_ids)
        x = tok + pos
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(batch_size * seq_len, -1),
                targets.view(batch_size * seq_len),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.ckpt, map_location=device)
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    model = TinyLM(
        vocab_size=len(stoi),
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if len(prompt_ids) == 0:
        prompt_ids = [0]

    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    output_ids = model.generate(
        start,
        max_new_tokens=args.gen_len,
        temperature=args.temperature,
    )[0].tolist()

    print("===== inference output =====")
    print(decode(output_ids, itos))


if __name__ == "__main__":
    main()
