import argparse
from pathlib import Path
import torch

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
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    stoi = {ch: i for i, ch in enumerate(chars)}  # 字符 -> id
    itos = {i: ch for ch, i in stoi.items()}      # id -> 字符
    return stoi, itos


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    text = load_text(args.data)
    stoi, itos = build_vocab(text)

    print("==== MiniGPT Step2 ====")
    print("device:", device)
    print("text length:", len(text))
    print("vocab size:", len(stoi))
    print("first 20 chars in vocab:", list(stoi.keys())[:20])


if __name__ == "__main__":
    main()
