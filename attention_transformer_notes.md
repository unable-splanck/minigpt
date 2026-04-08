# MiniGPT Full GPT Notes

这份文档对应当前项目里的三个核心文件：

- [model.py](/Users/bahesplanck/minigpt/model.py)：完整 GPT 结构
- [train.py](/Users/bahesplanck/minigpt/train.py)：数据、训练、保存 checkpoint
- [inference.py](/Users/bahesplanck/minigpt/inference.py)：加载 checkpoint 并生成文本

这一版已经从“单头 attention 的教学版”升级成了一个更完整的 GPT 骨架：

- token embedding
- position embedding
- multi-head causal self-attention
- residual connection
- LayerNorm
- feed-forward network
- 多层 block 堆叠
- dropout
- 最终 LayerNorm
- 自回归生成

本文用“数学公式 + 代码对应 + 直觉解释”的方式说明它是怎么工作的。

## 1. 总体结构

当前模型主体在 [model.py](/Users/bahesplanck/minigpt/model.py) 的 `MiniGPT` 类。

它的计算图可以写成：

$$
X_0 = E(\text{idx}) + P
$$

$$
X_{\ell} = \text{Block}_{\ell}(X_{\ell-1}), \quad \ell = 1,2,\dots,L
$$

$$
H = \text{LayerNorm}(X_L)
$$

$$
\text{logits} = HW_{vocab}
$$

这里：

- $E(\text{idx})$ 是 token embedding
- $P$ 是 position embedding
- $L$ 是 Transformer block 的层数
- `logits` 是对词表中每个 token 的未归一化分数

对应代码主干：

```python
tok = self.token_emb(idx)
pos = self.pos_emb(pos_ids)
x = self.drop(tok + pos)

for block in self.blocks:
    x = block(x)

x = self.ln_f(x)
logits = self.lm_head(x)
```

## 2. 输入为什么要做 Embedding

原始输入是 token id，例如：

```text
[12, 5, 31, 7]
```

这些数字只是索引，不具备可计算的语义空间。  
所以我们先用 embedding 把它变成连续向量。

如果：

- batch size 为 $B$
- 序列长度为 $T$
- 隐层维度为 $C$

那么：

$$
\text{idx} \in \mathbb{R}^{B \times T}
$$

经过 token embedding 后：

$$
E(\text{idx}) \in \mathbb{R}^{B \times T \times C}
$$

position embedding：

$$
P \in \mathbb{R}^{T \times C}
$$

最终输入：

$$
X_0 = E(\text{idx}) + P
$$

对应代码在 [model.py](/Users/bahesplanck/minigpt/model.py) 的 `MiniGPT.forward()`。

直觉理解：

- token embedding 负责“这是什么 token”
- position embedding 负责“它在第几个位置”

## 3. GPT 的核心：Causal Self-Attention

### 3.1 Query / Key / Value

对于输入表示 $X$，每个 attention head 都会做三次线性投影：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

对应代码在 [model.py](/Users/bahesplanck/minigpt/model.py) 的 `CausalSelfAttention`：

```python
self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
```

可以把它们理解成：

- `Q`：当前位置“想找什么”
- `K`：每个位置“能提供什么特征”
- `V`：真正被聚合的内容

### 3.2 多头注意力

如果总维度是 $C$，head 数量是 $h$，那么每个 head 的维度是：

$$
d_h = \frac{C}{h}
$$

代码里会先把张量 reshape 成：

$$
Q, K, V \in \mathbb{R}^{B \times h \times T \times d_h}
$$

对应代码：

```python
q = self.query(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
k = self.key(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
v = self.value(x).view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
```

多头注意力的意义是：  
不同 head 可以学习不同类型的关系，例如局部模式、长距离依赖、语法结构等。

### 3.3 缩放点积注意力

每个 head 的分数矩阵为：

$$
S = \frac{QK^T}{\sqrt{d_h}}
$$

对应代码：

```python
att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
```

除以 $\sqrt{d_h}$ 的原因是为了防止维度变大时点积数值过大，导致 `softmax` 过于尖锐。

### 3.4 Causal Mask

GPT 是自回归模型，所以位置 $t$ 只能看自己和前面的 token，不能看未来。

数学上：

$$
\tilde{S}_{ij} =
\begin{cases}
S_{ij}, & j \le i \\
-\infty, & j > i
\end{cases}
$$

对应代码：

```python
att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
```

这里的 `mask` 是一个下三角矩阵。

### 3.5 Softmax 与加权求和

注意力权重：

$$
A = \text{softmax}(\tilde{S})
$$

最终输出：

$$
O = AV
$$

对应代码：

```python
att = F.softmax(att, dim=-1)
y = att @ v
```

每个位置现在都能根据权重，从历史 token 中聚合信息。

### 3.6 多头拼接与输出投影

多头结果会先拼回原始维度，再做输出投影：

$$
\text{MultiHead}(X) = \text{Concat}(O_1, O_2, \dots, O_h)W_O
$$

更细一点写，可以分成两步：

第一步，把每个 head 的输出沿最后一维拼接：

$$
O_{\text{cat}} = \text{Concat}(O_1, O_2, \dots, O_h)
$$

如果：

$$
O_i \in \mathbb{R}^{B \times T \times d_h}
$$

那么拼接之后：

$$
O_{\text{cat}} \in \mathbb{R}^{B \times T \times (h \cdot d_h)} = \mathbb{R}^{B \times T \times C}
$$

第二步，再经过输出投影矩阵：

$$
Y = O_{\text{cat}} W_O + b_O
$$

其中：

$$
W_O \in \mathbb{R}^{C \times C}
$$

因此最终：

$$
Y \in \mathbb{R}^{B \times T \times C}
$$

这里的 `Y` 就是 attention 模块真正返回给 Transformer 主干的结果。

可以把这一步理解成：

- `O_1, O_2, \dots, O_h`：每个 head 分别提取出的上下文特征
- `O_cat`：把这些特征并排拼起来
- `Y`：再经过一次线性混合，让不同 head 的信息重新融合，回到主干隐藏空间

如果没有这一步投影，虽然多头各自都算出了上下文信息，但这些 head 的输出只是“机械拼接”，还没有被统一整合成主干后续 block 更容易处理的表示。

对应代码：

```python
y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
y = self.proj(y)
```

这两行正好对应上面的两步：

1. `transpose(...).view(...)`  
把原来形状为：

$$
\mathbb{R}^{B \times h \times T \times d_h}
$$

的张量，重新排成：

$$
O_{\text{cat}} \in \mathbb{R}^{B \times T \times C}
$$

2. `self.proj(y)`  
执行：

$$
Y = O_{\text{cat}} W_O + b_O
$$

输出最终的 attention 表示：

$$
Y \in \mathbb{R}^{B \times T \times C}
$$

### 3.7 多头注意力里的张量数据流

这一段是最容易“代码能看懂，但脑子里没有图像”的地方。  
我们把 [model.py](/Users/bahesplanck/minigpt/model.py:38) 到 [model.py](/Users/bahesplanck/minigpt/model.py:59) 的数据流完整拆开。

假设当前配置是：

- $B = 8$：batch size
- $T = 64$：序列长度
- $C = 128$：embedding 维度
- $h = 4$：注意力头数

那么每个 head 的维度是：

$$
d_h = C / h = 32
$$

输入到 attention 的张量：

$$
X \in \mathbb{R}^{8 \times 64 \times 128}
$$

第一步，线性映射得到：

$$
Q, K, V \in \mathbb{R}^{8 \times 64 \times 128}
$$

也就是：每个位置都被投影成三套不同语义的表示。

第二步，reshape + transpose：

$$
Q, K, V \in \mathbb{R}^{8 \times 4 \times 64 \times 32}
$$

这一步的含义不是“信息变多了”，而是把原来的 128 维拆成 4 个 head，每个 head 只处理 32 维子空间。

第三步，计算 attention score：

$$
S = \frac{QK^T}{\sqrt{32}}
$$

此时：

$$
S \in \mathbb{R}^{8 \times 4 \times 64 \times 64}
$$

可以把它理解成：

- 对每个 batch
- 对每个 head
- 都得到一个 `64 x 64` 的注意力分数矩阵

其中第 `i` 行、第 `j` 列表示：

“当前位置 `i` 对历史位置 `j` 的关注强度”

第四步，mask + softmax 之后：

$$
A \in \mathbb{R}^{8 \times 4 \times 64 \times 64}
$$

这时候每一行已经变成概率分布，表示“当前 token 应该把多少注意力分给哪些历史位置”。

第五步，和 $V$ 相乘：

$$
O = AV \in \mathbb{R}^{8 \times 4 \times 64 \times 32}
$$

也就是说，每个 head 都为每个位置生成了一个新的 32 维上下文表示。

第六步，把 4 个 head 拼接回去：

$$
\text{Concat}(O_1, O_2, O_3, O_4) \in \mathbb{R}^{8 \times 64 \times 128}
$$

最后再经过输出投影：

$$
Y \in \mathbb{R}^{8 \times 64 \times 128}
$$

这就是为什么多头注意力最终既能“并行从多个角度看上下文”，又能把结果重新对齐回模型主干维度。

### 3.8 为什么多头比单头强

单头注意力的问题不是“不能工作”，而是它只能用一种方式解释上下文关系。

多头注意力更强，是因为它允许模型同时学习多种关系：

- 一个 head 关注近邻 token
- 一个 head 关注更远的长程依赖
- 一个 head 关注标点或分隔结构
- 一个 head 关注局部语义组合

虽然这些 head 没有人手工指定语义，但训练后常常会形成这种分工。

如果把单头 attention 比作“一个人看一篇文章”，  
那多头 attention 更像“几个人分别从不同角度做笔记，然后把结果汇总”。

## 4. Transformer Block

一个标准 GPT block 做两件事：

1. attention
2. feed-forward

当前代码使用的是 pre-LN 结构：

$$
X' = X + \text{Attention}(\text{LN}(X))
$$

$$
Y = X' + \text{FFN}(\text{LN}(X'))
$$

对应代码在 [model.py](/Users/bahesplanck/minigpt/model.py) 的 `Block.forward()`：

```python
x = x + self.attn(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

### 4.1 为什么要有残差连接

残差连接的作用：

- 保留原始信息
- 让梯度更容易传播
- 让更深的网络更稳定

如果没有残差，深层模型常常更难训练。

### 4.2 为什么要有 LayerNorm

`LayerNorm` 会把每个位置上的隐藏表示做归一化，帮助训练时的数值分布保持稳定。

它的作用可以理解成：

- 降低数值漂移
- 提高训练稳定性
- 让 attention 和 FFN 的输入尺度更可控

### 4.3 一个 Block 里的完整数据流

假设 block 的输入是：

$$
X \in \mathbb{R}^{B \times T \times C}
$$

那么在一个 block 里，数据按这个顺序流动：

1. `ln1(x)`  
输入先归一化，形状不变：

$$
\text{LN}(X) \in \mathbb{R}^{B \times T \times C}
$$

2. `attn(ln1(x))`  
attention 聚合历史信息，输出仍然是：

$$
\text{Attn}(\text{LN}(X)) \in \mathbb{R}^{B \times T \times C}
$$

3. `x + ...`  
和原始输入相加，得到第一条残差路径结果：

$$
X' = X + \text{Attn}(\text{LN}(X))
$$

4. `ln2(x')`  
再次归一化：

$$
\text{LN}(X') \in \mathbb{R}^{B \times T \times C}
$$

5. `ffwd(ln2(x'))`  
逐位置做非线性变换，输出仍然保持主干维度：

$$
\text{FFN}(\text{LN}(X')) \in \mathbb{R}^{B \times T \times C}
$$

6. 再做一次残差相加：

$$
Y = X' + \text{FFN}(\text{LN}(X'))
$$

最后：

$$
Y \in \mathbb{R}^{B \times T \times C}
$$

注意这个结论非常重要：  
Transformer block 虽然内部很复杂，但它对外始终保持“输入什么形状，输出还是什么形状”。  
这也是它能被反复堆叠的根本原因。

## 5. Feed-Forward Network

每个 block 里除了 attention，还有一个前馈网络：

$$
\text{FFN}(x) = W_2 \sigma(W_1x + b_1) + b_2
$$

在代码里：

```python
self.net = nn.Sequential(
    nn.Linear(config.n_embd, 4 * config.n_embd),
    nn.GELU(),
    nn.Linear(4 * config.n_embd, config.n_embd),
    nn.Dropout(config.dropout),
)
```

可以把它理解成：

- attention 决定“看谁”
- FFN 决定“看完之后怎么加工表示”

## 6. 多层堆叠为什么重要

一个 block 只能做一层信息混合。  
GPT 真正强大的地方在于它可以堆很多层：

$$
X_1 = \text{Block}_1(X_0)
$$

$$
X_2 = \text{Block}_2(X_1)
$$

$$
\dots
$$

$$
X_L = \text{Block}_L(X_{L-1})
$$

对应代码：

```python
self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
for block in self.blocks:
    x = block(x)
```

层数越多，模型越能逐层提取更复杂的模式。

## 7. Dropout 在哪里用

当前 GPT 结构里 dropout 用在两个地方：

1. embedding 之后
2. attention 和 FFN 输出之后

作用是正则化，降低过拟合风险。

对应代码：

```python
self.drop = nn.Dropout(config.dropout)
self.attn_dropout = nn.Dropout(config.dropout)
self.resid_dropout = nn.Dropout(config.dropout)
```

## 8. 输出层与 Weight Tying

最后模型输出：

$$
\text{logits} = HW_{vocab}
$$

在代码里：

```python
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
self.lm_head.weight = self.token_emb.weight
```

这里做了 `weight tying`，也就是输出层权重和输入 embedding 权重共享。  
这在 GPT 类模型里是常见做法，可以减少参数量，并让输入输出空间更一致。

## 9. Loss：为什么是 Next-Token Prediction

训练目标是：

“给定前文，预测下一个 token”

如果输入是：

```text
x = [a, b, c]
```

目标就是：

```text
y = [b, c, d]
```

损失函数：

$$
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i \mid x_{\le i})
$$

对应代码在 [model.py](/Users/bahesplanck/minigpt/model.py)：

```python
loss = F.cross_entropy(
    logits.view(batch_size * seq_len, -1),
    targets.view(batch_size * seq_len),
)
```

## 10. 训练脚本在做什么

[train.py](/Users/bahesplanck/minigpt/train.py) 负责的是工程流程，而不是模型结构本身。

它做了这些事：

1. 读取文本
2. 构建字符词表
3. 编码成 token id
4. 随机采样 batch
5. 调用 `MiniGPT.forward()`
6. 反向传播并更新参数
7. 保存 checkpoint
8. 用 prompt 做一次生成测试

其中 `get_batch()` 很关键，因为它构造了语言模型的监督信号：

```python
x = data[i : i + block_size]
y = data[i + 1 : i + block_size + 1]
```

这就是“输入当前序列，预测右移一位后的目标序列”。

## 11. 从数据文件到 loss 的完整数据流

这一节专门把 [train.py](/Users/bahesplanck/minigpt/train.py) 中的数据流按执行顺序串起来。

### 11.1 读文件

在 [train.py](/Users/bahesplanck/minigpt/train.py:37) 之后，脚本从 `data.txt` 读取原始文本。

例如：

```text
Transformer models learn from next-token prediction.
```

### 11.2 构建词表

在 [train.py](/Users/bahesplanck/minigpt/train.py:52) 附近，脚本会把文本里的所有字符去重，得到：

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
```

这里：

- `stoi`：字符 -> 整数 id
- `itos`：整数 id -> 字符

### 11.3 文本编码

在 [train.py](/Users/bahesplanck/minigpt/train.py:58) 附近：

```python
data = encode(text, stoi)
```

原始字符串会变成一维 token id 序列：

$$
\text{data} \in \mathbb{R}^{N}
$$

这里的 $N$ 是整份语料的字符数。

### 11.4 随机采样 batch

`get_batch()` 从整份长序列里随机切片，得到：

$$
x \in \mathbb{R}^{B \times T}, \quad y \in \mathbb{R}^{B \times T}
$$

其中：

- `x` 是输入片段
- `y` 是右移一位的目标片段

例如：

```text
x = "Transfor"
y = "ransform"
```

### 11.5 模型前向

`x` 进入 [model.py](/Users/bahesplanck/minigpt/model.py:112) 的 `MiniGPT.forward()` 后，会依次经过：

1. token embedding
2. position embedding
3. dropout
4. 多层 Transformer block
5. 最终 LayerNorm
6. 词表线性映射

得到：

$$
\text{logits} \in \mathbb{R}^{B \times T \times V}
$$

### 11.6 计算 loss

然后会把 `logits` reshape 成：

$$
\mathbb{R}^{(B \cdot T) \times V}
$$

把目标 `y` reshape 成：

$$
\mathbb{R}^{B \cdot T}
$$

这样就能用 `cross_entropy` 在“所有位置上的 next-token 分类任务”上一起算平均损失。

### 11.7 反向传播和参数更新

在 [train.py](/Users/bahesplanck/minigpt/train.py:108) 之后：

```python
loss.backward()
optimizer.step()
```

这意味着：

- `loss.backward()` 计算所有参数的梯度
- `optimizer.step()` 用梯度更新参数

整个训练流程的目标，就是不断降低：

$$
\mathcal{L} = - \log p_\theta(y \mid x)
$$

## 12. 从 prompt 到生成文本的数据流

推理阶段的数据流和训练不同，因为这时没有标签 `y`，只有 prompt。

### 12.1 prompt 编码

在 [inference.py](/Users/bahesplanck/minigpt/inference.py:36) 附近：

```python
prompt_ids = [stoi[ch] for ch in args.prompt if ch in stoi]
start = torch.tensor([prompt_ids], dtype=torch.long, device=device)
```

于是：

$$
\text{start} \in \mathbb{R}^{1 \times T_0}
$$

这里 $T_0$ 是 prompt 长度。

### 12.2 截断上下文窗口

在 [model.py](/Users/bahesplanck/minigpt/model.py:139)：

```python
idx_cond = idx[:, -self.config.block_size :]
```

这表示如果当前上下文太长，只保留最后 `block_size` 个 token。  
因为模型训练时只学过这个窗口大小。

### 12.3 只取最后一个位置的分布

模型前向得到：

$$
\text{logits} \in \mathbb{R}^{1 \times T \times V}
$$

但生成时只关心最后一个位置：

```python
logits = logits[:, -1, :]
```

也就是：

$$
\text{logits}_{last} \in \mathbb{R}^{1 \times V}
$$

### 12.4 temperature 和 top-k

先做温度缩放：

$$
\text{logits}' = \frac{\text{logits}}{\tau}
$$

其中：

- $\tau < 1$：分布更尖锐，输出更保守
- $\tau > 1$：分布更平缓，输出更随机

再做 top-k：

只保留概率最高的前 $k$ 个 token，其余位置直接置为 $-\infty$。

这样可以显著减少“非常低概率但被采样到”的噪声 token。

### 12.5 采样并拼回序列

最后：

```python
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)
idx = torch.cat([idx, next_id], dim=1)
```

于是长度从：

$$
1 \times T
$$

变成：

$$
1 \times (T+1)
$$

接着重复这个过程，直到生成完 `max_new_tokens`。

## 13. 推理脚本在做什么

[inference.py](/Users/bahesplanck/minigpt/inference.py) 会：

1. 读取 checkpoint
2. 恢复 `GPTConfig`
3. 初始化 `MiniGPT`
4. 加载权重
5. 根据 prompt 自回归生成文本

这意味着训练和推理现在共享同一个模型定义，不会再出现“训练结构和推理结构对不上”的问题。

## 14. 自回归生成的数学形式

生成时，每一步只预测一个新 token：

$$
x_{t+1} \sim p_\theta(\cdot \mid x_1, x_2, \dots, x_t)
$$

对应代码：

```python
logits, _ = self(idx_cond)
logits = logits[:, -1, :]
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)
idx = torch.cat([idx, next_id], dim=1)
```

这里还支持：

- `temperature`：控制随机性
- `top_k`：只在概率最高的前 `k` 个 token 中采样

## 15. 这版代码和“教学版”相比，真正升级了什么

之前的版本主要是：

- embedding
- 单头 attention
- 一个 block

现在这版已经升级成：

- 多头 attention
- 残差连接
- LayerNorm
- FFN
- 多层 block
- dropout
- 配置化 `GPTConfig`
- 共享模型定义的训练与推理流程

这已经是一个很标准的 MiniGPT 骨架了。

## 16. 你接下来最值得观察的现象

训练这版 GPT 时，你可以重点看三件事：

1. `loss` 是否比旧版本下降更稳定
2. 生成文本是否更像连续句子，而不是纯字符噪声
3. 当你调整 `n_layer / n_head / n_embd / block_size` 时，输出风格如何变化

## 17. 一句话总结

完整 GPT 不是“多堆几层线性层”，而是：

通过多头自注意力决定“看谁”，  
通过 FFN 决定“怎么加工”，  
通过残差和 LayerNorm 保证这些层能稳定堆起来，  
最后用 next-token prediction 把整个系统训练成一个自回归语言模型。
