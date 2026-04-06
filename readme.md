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
- `--n-layer` Transformer 层数
- `--n-head` 注意力头数
- `--n-embd` 隐层维度
- `--lr` 学习率
- `--out` 模型保存路径

## 4. 建议的项目迭代方向

1. 从字符级改成 BPE/WordPiece 分词
2. 增加学习率 warmup + cosine decay
3. 支持混合精度训练（AMP）
4. 增加独立 `inference.py`，加载 checkpoint 做交互式生成
5. 做一个简单 Web Demo（FastAPI + 前端）

---

如果你愿意，我下一步可以继续帮你加上：
- `inference.py`（加载 `minigpt.pt` 单独推理）
- `requirements.txt`
- 一个可直接放到简历里的项目说明模板
