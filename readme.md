# MiniGPT 项目（PyTorch）

这是一个从零实现的 **字符级 MiniGPT** 训练脚本，适合做大模型入门项目与作品集。

## 1. 环境准备

```bash
cd /Users/bahesplanck/minigpt
source .venv/bin/activate
```

安装依赖（如果尚未安装）：

```bash
pip install torch
```

## 2. 快速开始（无需数据文件）

直接使用脚本内置示例语料：

```bash
python train.py
```

你会看到：
- 训练/验证 loss
- best checkpoint 保存路径（默认 `minigpt.pt`）
- 采样生成文本

## 3. 使用你自己的语料

准备一个 `.txt` 文件（UTF-8），例如 `data.txt`：

```bash
python train.py --data data.txt --max-steps 2000 --batch-size 32 --block-size 128
```

常用参数：
- `--lr` 学习率
- `--out` 模型保存路径
- `--prompt` 训练结束后用于生成的起始文本
- `--gen-len` 生成长度
- `--temperature` 采样温度

## 4. 单独推理

训练完成后，可以直接加载保存的 checkpoint 生成文本：

```bash
python inference.py --ckpt minigpt.pt --prompt "MiniGPT " --gen-len 120
```

## 5. 建议的项目迭代方向

1. 从字符级改成 BPE/WordPiece 分词
2. 增加学习率 warmup + cosine decay
3. 支持混合精度训练（AMP）
4. 给 `inference.py` 增加 `top-k` 和 `top-p` 采样
5. 做一个简单 Web Demo（FastAPI + 前端）

---
