# MiniGPT 项目（PyTorch）

这是一个从零实现的 `MiniGPT` 项目，当前已经包含一套完整的 GPT 骨架：

- 多头 `causal self-attention`
- `LayerNorm + Residual + FeedForward`
- 多层 `Transformer Block`
- dropout
- checkpoint 保存与加载
- 独立推理脚本
- 配套原理文档

项目里的核心文件：

- [model.py](/minigpt/model.py)：GPT 模型定义
- [train.py](/minigpt/train.py)：训练脚本
- [inference.py](/minigpt/inference.py)：推理脚本
- [clean_data.py](/minigpt/clean_data.py)：简单数据清洗脚本
- [attention_transformer_notes.md](/minigpt/attention_transformer_notes.md)：公式结合代码的详细说明

## 1. 环境准备

```bash
cd ./minigpt
source .venv/bin/activate
```

如果环境里还没有安装 `torch`：

```bash
pip install torch
```

## 2. 数据准备

项目里已经有一份示例语料：[data.txt](/minigpt/data.txt)

如果你想先做一次简单清洗：

```bash
python clean_data.py --input data.txt --output data_clean.txt
```

然后训练时可以直接使用清洗后的文件：

```bash
python train.py --data data_clean.txt
```

如果不传 `--data`，训练脚本会退回到 `train.py` 内置的 `DEFAULT_CORPUS`。

## 3. 训练模型

最简单的训练方式：

```bash
python train.py --data data.txt
```

一个更推荐的训练命令：

```bash
python train.py \
  --data data.txt \
  --max-steps 3000 \
  --batch-size 32 \
  --block-size 128 \
  --n-layer 4 \
  --n-head 4 \
  --n-embd 128 \
  --lr 3e-4 \
  --out minigpt.pt
```

训练时会输出：

- 当前设备
- 数据长度和词表大小
- 模型配置
- 训练过程中的 loss
- 最终 loss
- 保存的 checkpoint 路径
- 训练结束后的示例生成结果

常用参数：

- `--data`：训练语料路径
- `--batch-size`：batch 大小
- `--block-size`：上下文窗口长度
- `--max-steps`：训练步数
- `--lr`：学习率
- `--n-layer`：Transformer block 层数
- `--n-head`：注意力头数
- `--n-embd`：隐藏维度
- `--dropout`：dropout 概率
- `--out`：checkpoint 保存路径
- `--prompt`：训练结束后用于测试生成的起始 prompt
- `--gen-len`：生成长度
- `--temperature`：采样温度
- `--top-k`：只在前 `k` 个候选 token 中采样

## 4. 单独推理

训练完成后，可以直接加载 checkpoint 生成文本：

```bash
python inference.py --ckpt minigpt.pt --prompt "Transformer " --gen-len 120
```

如果你想在运行时手动输入 prompt：

```bash
python inference.py --ckpt minigpt.pt
```

脚本会在终端显示：

```bash
prompt>
```

推理常用参数：

- `--ckpt`：checkpoint 路径
- `--prompt`：起始 prompt；不传时会进入交互输入
- `--gen-len`：生成长度
- `--temperature`：采样温度
- `--top-k`：top-k 采样

## 5. 模型结构

当前 `MiniGPT` 的整体流程是：

```text
token ids
-> token embedding + position embedding
-> dropout
-> multi-head causal self-attention
-> residual + LayerNorm + FeedForward
-> stacked Transformer blocks
-> final LayerNorm
-> lm_head
-> logits
```

更详细的数学公式、张量形状变化和数据流说明，请看：

- [attention_transformer_notes.md](/minigpt/attention_transformer_notes.md)

## 6. 当前项目适合做什么

这个项目比较适合：

- 学习 GPT 的基本结构和代码实现
- 理解自注意力、多头注意力、残差和 LayerNorm
- 练习从数据到训练再到推理的完整闭环
- 做一个可解释、可继续扩展的大模型入门项目

## 7. 后续可扩展方向

1. 增加 `train / val split` 和验证集 loss
2. 加学习率 warmup 和 cosine decay
3. 支持混合精度训练
4. 从字符级改成子词级 tokenization
5. 增加 `top-p` 采样
6. 做一个简单的 Web Demo 或 API 服务
