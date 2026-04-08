# MiniGPT Self-Attention And Transformer Notes

这份文档结合当前项目里的 [train.py](/Users/bahesplanck/minigpt/train.py) 来解释两件事：

1. 代码里已经实现的 `SelfAttentionHead` 到底在做什么。
2. 它和完整 Transformer 的关系是什么。

本文尽量用“数学公式 + 代码位置 + 直觉解释”的方式来写。

## 1. 先看整体数据流

当前模型的主干在 [train.py](/Users/bahesplanck/minigpt/train.py:113)。

它的计算流程可以概括成：

$$
\text{token ids} \rightarrow \text{token embedding} + \text{position embedding}
\rightarrow \text{self-attention}
\rightarrow \text{linear head}
\rightarrow \text{logits}
\rightarrow \text{cross entropy loss}
$$

对应代码：

```python
tok = self.token_emb(idx)
pos_ids = torch.arange(T, device=idx.device)
pos = self.pos_emb(pos_ids)
x = tok + pos
x = self.sa_head(x)
logits = self.lm_head(x)
```

这里的关键思想是：

- `idx` 是离散 token id，模型不能直接拿整数做复杂计算。
- `embedding` 把离散 id 变成连续向量。
- `self-attention` 让当前位置可以“看前文”。
- `lm_head` 把隐藏表示映射回词表大小，输出每个 token 的分数。

## 2. 输入表示：Token Embedding 和 Position Embedding

在 [train.py](/Users/bahesplanck/minigpt/train.py:117) 和 [train.py](/Users/bahesplanck/minigpt/train.py:118)：

```python
self.token_emb = nn.Embedding(vocab_size, n_embd)
self.pos_emb = nn.Embedding(block_size, n_embd)
```

假设：

- batch 大小为 $B$
- 序列长度为 $T$
- embedding 维度为 $C$
- 词表大小为 $V$

那么：

$$
\text{idx} \in \mathbb{R}^{B \times T}
$$

经过 token embedding 之后：

$$
E(\text{idx}) \in \mathbb{R}^{B \times T \times C}
$$

位置 embedding：

$$
P \in \mathbb{R}^{T \times C}
$$

最后输入到 attention 的表示为：

$$
X = E(\text{idx}) + P
$$

对应代码：

```python
tok = self.token_emb(idx)   # [B, T, C]
pos = self.pos_emb(pos_ids) # [T, C]
x = tok + pos               # [B, T, C]
```

直觉上可以理解为：

- `token_emb` 告诉模型“这个字符/词是什么”
- `pos_emb` 告诉模型“它在第几个位置”

如果没有位置编码，模型只知道有哪些 token，却不知道顺序。

## 3. Self-Attention 的核心目标

自注意力要解决的问题是：

“当前第 $t$ 个位置在预测下一个 token 时，应该参考前面哪些位置？”

例如句子：

```text
machine learning is ...
```

当模型看到 `is` 后面要继续生成时，它可能需要回看 `machine` 和 `learning`。

传统线性层不会显式做这件事，而 self-attention 会为每个位置计算“应该关注谁”。

## 4. Query / Key / Value 是什么

在 [train.py](/Users/bahesplanck/minigpt/train.py:85) 到 [train.py](/Users/bahesplanck/minigpt/train.py:87)：

```python
self.key = nn.Linear(n_embd, head_size, bias=False)
self.query = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

对于输入表示 $X \in \mathbb{R}^{B \times T \times C}$，我们做三个线性投影：

$$
Q = XW_Q
$$

$$
K = XW_K
$$

$$
V = XW_V
$$

其中：

- $Q$ 表示“我现在想找什么信息”
- $K$ 表示“我这个位置能提供什么信息”
- $V$ 表示“我真正要传递出去的内容”

对应代码：

```python
k = self.key(x)
q = self.query(x)
v = self.value(x)
```

为什么需要三套矩阵，而不是一套？

因为“拿什么去匹配别人”和“最终传什么内容”不一定相同。  
这是 attention 能灵活表达上下文关系的重要原因。

## 5. 注意力分数是怎么计算的

在 [train.py](/Users/bahesplanck/minigpt/train.py:100)：

```python
wei = q @ k.transpose(-2, -1) * (C ** -0.5)
```

数学形式是：

$$
S = \frac{QK^T}{\sqrt{d_k}}
$$

这里：

- $S \in \mathbb{R}^{B \times T \times T}$
- $S_{ij}$ 表示第 $i$ 个位置对第 $j$ 个位置的关注分数

为什么要除以 $\sqrt{d_k}$？

因为维度变大时，点积的数值会变大，`softmax` 容易过于尖锐，训练不稳定。  
缩放之后，数值范围更平稳，这也是 Transformer 原论文里的标准写法。

## 6. Causal Mask 为什么必须有

在语言模型里，第 $t$ 个位置只能看见自己和前面的位置，不能偷看未来 token。

在 [train.py](/Users/bahesplanck/minigpt/train.py:90)：

```python
self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
```

这里构造了一个下三角矩阵：

$$
M =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

在 [train.py](/Users/bahesplanck/minigpt/train.py:103)：

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

意思是把未来位置的分数强行设成 $-\infty$。  
这样经过 `softmax` 之后，这些位置的概率就会变成 0。

数学上可以写成：

$$
\tilde{S}_{ij} =
\begin{cases}
S_{ij}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

这一步是自回归语言模型成立的关键。

## 7. Softmax：把分数变成注意力权重

在 [train.py](/Users/bahesplanck/minigpt/train.py:106)：

```python
wei = F.softmax(wei, dim=-1)
```

数学形式：

$$
A = \text{softmax}(\tilde{S})
$$

每一行满足：

$$
\sum_j A_{ij} = 1
$$

这时 $A_{ij}$ 可以理解为：

“位置 $i$ 在聚合信息时，对位置 $j$ 分配了多少注意力。”

## 8. 用 Value 做加权求和

在 [train.py](/Users/bahesplanck/minigpt/train.py:108) 和 [train.py](/Users/bahesplanck/minigpt/train.py:109)：

```python
v = self.value(x)
out = wei @ v
```

数学形式：

$$
O = AV
$$

其中：

$$
O \in \mathbb{R}^{B \times T \times C}
$$

直觉上：

- `A` 决定“看谁”
- `V` 决定“拿到什么信息”
- `O` 是聚合后的上下文表示

于是，每个位置不再只依赖自己的 embedding，而是变成了“结合历史上下文后的表示”。

## 9. 当前代码里的 Self-Attention 是单头注意力

你现在的实现只有一个头，也就是：

```python
self.sa_head = SelfAttentionHead(n_embd, block_size)
```

这意味着整个表示空间只用一种方式去看上下文。

完整的多头注意力通常会写成：

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W_O
$$

多头的意义是：

- 有的头学语法关系
- 有的头学局部邻近关系
- 有的头学长距离依赖

单头能帮助你理解机制，但表达能力有限。

## 10. 当前模型如何得到 logits

Self-Attention 输出之后，代码在 [train.py](/Users/bahesplanck/minigpt/train.py:134)：

```python
logits = self.lm_head(x)
```

数学形式：

$$
\text{logits} = OW_{lm} + b
$$

输出张量形状为：

$$
\text{logits} \in \mathbb{R}^{B \times T \times V}
$$

意思是：

每个 batch、每个位置，模型都会对整个词表中的每个 token 给出一个分数。

这里的 `logits` 还不是概率。  
如果对最后一维做 `softmax`，才会得到真正的概率分布。

## 11. Loss 是怎样和语言建模对应起来的

在 [train.py](/Users/bahesplanck/minigpt/train.py:139) 到 [train.py](/Users/bahesplanck/minigpt/train.py:142)：

```python
loss = F.cross_entropy(
    logits.view(B * T, -1),
    targets.view(B * T)
)
```

这里做的是 next-token prediction，也就是：

“给定前文，预测下一个 token 是什么。”

如果输入是：

```text
x = [a, b, c]
```

那么目标可能是：

```text
y = [b, c, d]
```

数学上，交叉熵损失可以写成：

$$
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i | x_{\le i})
$$

这说明模型在每个位置都要尽量把真实下一个 token 的概率拉高。

## 12. 生成阶段为什么是一 token 一 token 地采样

在 [train.py](/Users/bahesplanck/minigpt/train.py:145) 之后的 `generate()`：

```python
logits, _ = self(idx_cond)
logits = logits[:, -1, :]
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, 1)
idx = torch.cat([idx, next_id], dim=1)
```

生成过程本质上是：

1. 给模型一个当前上下文
2. 只取最后一个位置的输出分布
3. 从这个分布里采样一个新 token
4. 把新 token 接回输入
5. 重复以上过程

数学形式：

$$
x_{t+1} \sim p_\theta(\cdot \mid x_1, x_2, \ldots, x_t)
$$

这就是自回归生成。

## 13. 当前代码和“完整 Transformer”之间还差什么

虽然你已经实现了 Transformer 最核心的 `self-attention`，但当前 [train.py](/Users/bahesplanck/minigpt/train.py) 还不是完整的标准 Transformer block。

标准 block 往往是：

$$
X' = X + \text{Attention}(\text{LayerNorm}(X))
$$

$$
Y = X' + \text{MLP}(\text{LayerNorm}(X'))
$$

通常还会有：

- 多头注意力 `Multi-Head Attention`
- 残差连接 `Residual Connection`
- 层归一化 `LayerNorm`
- 前馈网络 `FeedForward / MLP`
- dropout

当前代码里已经有：

- token embedding
- position embedding
- 单头 causal self-attention
- 语言模型输出头
- 交叉熵训练
- 自回归采样生成

当前代码里还没有完整加上的部分：

- 多头注意力
- 残差连接
- LayerNorm
- MLP
- 多层 block 堆叠

这也是为什么它已经比最初的纯线性模型强，但仍然离“更稳定、更像正常句子”的 MiniGPT 有距离。

## 14. 一句话理解 Transformer

如果只用一句话概括：

Transformer 就是在每一层里反复做两件事：

1. 通过 attention 决定“当前 token 应该看谁”
2. 通过 MLP 决定“看完之后怎么更新自己的表示”

再配合残差和归一化，让这些层可以稳定堆起来。

## 15. 你可以怎样继续读这份代码

一个推荐的阅读顺序是：

1. 先看 [train.py](/Users/bahesplanck/minigpt/train.py:123) 到 [train.py](/Users/bahesplanck/minigpt/train.py:143)，理解 `forward()` 的整体流程。
2. 再看 [train.py](/Users/bahesplanck/minigpt/train.py:92) 到 [train.py](/Users/bahesplanck/minigpt/train.py:110)，理解 attention 的矩阵计算。
3. 最后回到 [train.py](/Users/bahesplanck/minigpt/train.py:185) 到 [train.py](/Users/bahesplanck/minigpt/train.py:192)，把训练和反向传播连接起来。

这样你会更容易把“代码执行顺序”和“公式里的变量流动”对应起来。

## 16. 推荐你接下来观察的现象

如果你边看文档边跑训练，建议重点观察三件事：

1. `loss` 是否下降  
这说明 attention 至少在帮助模型利用上下文。

2. 生成文本是否比纯线性模型更连贯  
这说明上下文聚合确实起作用了。

3. 当 `block_size` 增大时，生成是否更依赖长上下文  
这能帮助你直观理解“上下文窗口”的作用。

## 17. 最后总结

这次修改最重要的变化，不是“又多了一层线性变换”，而是：

模型第一次获得了一个明确机制，去动态决定“当前预测应该参考前面的哪些 token”。

这就是 self-attention 的本质。

而 Transformer，则是在 self-attention 的基础上，加上归一化、残差、前馈网络和多层堆叠，形成一个稳定、可扩展、能处理长上下文的架构。
