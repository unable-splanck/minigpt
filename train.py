import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# 使用方式
python train.py --data data.txt

'''
DEFAULT_CORPUS = (
    "miniGPT is a tiny transformer demo.\n"
    "we build it step by step.\n"
) * 50

# 参数设置
def parse_args():
    '''
    batch-size:
    block-size:
    max-step:
    lr:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out", type=str, default="minigpt.pt")
    parser.add_argument("--prompt", type=str, default="MiniGPT ")
    parser.add_argument("--gen-len", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    return parser.parse_args()


def load_text(data_path):
    # 未准备语料时使用
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
    stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 -> id
    itos = {i: ch for ch, i in stoi.items()}      # id -> 字符
    return stoi, itos


def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
def decode(token_ids, itos):
    return "".join([itos[i] for i in token_ids])


def get_batch(data, block_size, batch_size, device):
    # 随机选 batch_size 个起点
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    # x: [batch_size, block_size] 输入片段
    x = torch.stack([data[i:i + block_size] for i in ix])

    # y: 向右错一位，作为“下一个字符”的监督信号
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)

class SelfAttentionHead(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        head_size = n_embd

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 下三角 mask，防止看到未来 token
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        k = self.key(x)    # [B, T, C]
        q = self.query(x)  # [B, T, C]

        # 注意力分数: 当前 token 和前面 token 的相关性
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)   # [B, T, T]

        # causal mask: 未来位置设成 -inf
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # 转成概率
        wei = F.softmax(wei, dim=-1)   # [B, T, T]

        v = self.value(x)              # [B, T, C]
        out = wei @ v                  # [B, T, C]
        return out


class TinyLM(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd=64):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)   # token -> vector
        self.pos_emb = nn.Embedding(block_size, n_embd)     # position -> vector

        self.sa_head = SelfAttentionHead(n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)        # vector -> vocab logits

    def forward(self, idx, targets=None):
        # idx: [B, T]
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("sequence length > block_size")

        tok = self.token_emb(idx)  # [B, T, C]
        pos_ids = torch.arange(T, device=idx.device)         # [T]
        pos = self.pos_emb(pos_ids)                          # [T, C]
        x = tok + pos                                        # 广播到 [B, T, C]
        x = self.sa_head(x)
        logits = self.lm_head(x)                             # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # cross_entropy 需要 [N, vocab] 和 [N]
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
    # idx: [B, T]
        for _ in range(max_new_tokens):
            # 只保留最后 block_size 长度，防止超长
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)              # [B, T, vocab]
            logits = logits[:, -1, :]               # 只取最后一个位置 [B, vocab]
            logits = logits / temperature           # 温度控制随机性

            probs = F.softmax(logits, dim=-1)       # 转概率
            next_id = torch.multinomial(probs, 1)   # 按概率采样下一个 token

            idx = torch.cat([idx, next_id], dim=1)  # 拼接到序列后面
        return idx



def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text = load_text(args.data)
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    model = TinyLM(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_embd=64
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("==== MiniGPT Step6 (train loop) ====")
    print("device:", device)
    print("data length:", len(data))
    print("vocab size:", len(stoi))

    for step in range(1, args.max_steps + 1):
        xb, yb = get_batch(data, args.block_size, args.batch_size, device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:4d} | loss {loss.item():.4f}")

    # 训练后再看一个 batch 的 loss
    xb, yb = get_batch(data, args.block_size, args.batch_size, device)
    _, final_loss = model(xb, yb)
    print("final loss:", float(final_loss.detach()))

    # 保存 checkpoint（模型参数 + 词表 + 配置）
    ckpt = {
        "model_state_dict": model.state_dict(),
        "stoi": stoi,
        "itos": itos,
        "block_size": args.block_size,
        "n_embd": 64,
    }
    torch.save(ckpt, args.out)
    print("checkpoint saved to:", args.out)

    # 生成阶段使用 eval 模式，避免 dropout 干扰
    model.eval()

    # 用 prompt 做起始上下文；遇到未登录字符自动跳过
    prompt_ids = [stoi[ch] for ch in args.prompt if ch in stoi]
    if len(prompt_ids) == 0:
        prompt_ids = [0]
    start = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    out_ids = model.generate(
        start,
        max_new_tokens=args.gen_len,
        temperature=args.temperature,
    )[0].tolist()
    out_text = decode(out_ids, itos)

    print("\n===== generated text =====")
    print(out_text)




if __name__ == "__main__":
    main()
