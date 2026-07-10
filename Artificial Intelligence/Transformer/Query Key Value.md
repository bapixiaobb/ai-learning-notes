#LanguageModeling #Transformer #Attention #Architecture

Query、Key、Value，简称 QKV，是 [[Self-Attention]] 中用来计算 token 之间信息交互的三组向量。

## 🧠 Core Idea

>[!note]
>在 attention 中，每个 token 会产生三个向量：
>
>- Query：我想找什么信息？
>- Key：我能被什么 query 匹配？
>- Value：如果我被关注，我实际提供什么信息？

可以粗略理解为：

| Vector | Intuition                            |
| ------ | ------------------------------------ |
| Query  | 当前 token 发出的“查询”，也就是我这个 token 想找什么信息 |
| Key    | 每个 token 提供的“索引 / 匹配信号”              |
| Value  | 每个 token 真正被取走的信息内容                  |

## 🧩 Where QKV Comes From

给定输入 hidden states：

```math
X \in \mathbb{R}^{T \times d_{\text{model}}}
```

attention 会用三个 learnable projection matrices 生成 Q、K、V：

```math
Q = XW_Q
```

```math
K = XW_K
```

```math
V = XW_V
```

其中：

- $W_Q$ 生成 queries；
- $W_K$ 生成 keys；
- $W_V$ 生成 values。

>[!note]
>Q、K、V 都来自同一个 input sequence 时，就是 [[Self-Attention]]。
>
>如果 Q 来自 decoder，K/V 来自 encoder outputs，就是 [[Cross-Attention]]。

## 🔍 How They Work

Attention 的核心计算是：

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
```

可以分成两步理解，$d_k$ 是 dimension of head。

### 1. Query 和 Key 决定 attention weights

```math
QK^\top
```

表示每个 query 和每个 key 的匹配程度。

如果第 $i$ 个 token 的 query 和第 $j$ 个 token 的 key 很匹配，那么第 $i$ 个 token 就会更多关注第 $j$ 个 token。

### 2. Attention weights 对 Value 加权求和

[[Softmax]] 后得到 attention weights，再乘以 $V$：

```math
\mathrm{weights} \cdot V
```

得到新的 token representation。

>[!important]
>Q 和 K 决定“看谁”；V 决定“拿什么信息”。

### 3. Optimization with [[GPU]] 

[[Flash Attention]] 可以把 $QK^\top \rightarrow \mathrm{softmax}\rightarrow AV$ 这整个 attention 过程做成一个 memory- efficient fused kernel，以解决 [[GPU Memory Bound]] 的问题。 

## 📐 Shape View

如果省略 batch 和 head dimension：

```math
X \in \mathbb{R}^{T \times d_{\text{model}}}
```

通常：

```math
Q,K \in \mathbb{R}^{T \times d_k}
```

```math
V \in \mathbb{R}^{T \times d_v}
```

attention score matrix 是：

```math
QK^\top \in \mathbb{R}^{T \times T}
```

输出是：

```math
\mathrm{Attention}(Q,K,V)
\in
\mathbb{R}^{T \times d_v}
```

>[!note]
>$T \times T$ 的 score matrix 表示每个 token 对每个 token 的关注程度。

## 🎭 In Causal Attention

在 [[Causal Attention]] 中，QKV 仍然照常计算：

```math
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
```

区别是 attention score 会加上 [[Causal Mask]]：

```math
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
+
M
\right)V
```

mask 会阻止第 $t$ 个 token attend to future tokens。

>[!note]
>Causal mask 不改变 Q、K、V 本身。
>
>它只改变哪些 query-key pairs 可以参与 attention。

## 📍 Relation to RoPE

[[Rotary Position Embedding]] 通常作用在 Q 和 K 上：

```math
Q \rightarrow Q_{\mathrm{rope}}
```

```math
K \rightarrow K_{\mathrm{rope}}
```

但通常不作用在 V 上。

原因是：

>[!note]
>Q 和 K 决定 attention score，因此 position information 应该影响“谁关注谁”。
>
>V 是被加权汇总的信息内容，通常不需要用 RoPE 旋转。

## 🚫 Common Confusions

### Q、K、V 不是三份不同的输入文本

它们通常是同一个 hidden state $X$ 经过三个不同 linear projections 得到的。

### Query 不是数据库里的 query

名字有类比意义，但在 Transformer 中，query 是一个 learned vector representation。

### Value 不是最终输出

Value 是被 attention weights 加权汇总的内容。  
attention output 后面通常还会经过 output projection、residual connection、MLP 等模块。

---

>[!summary] My Understanding
>Query、Key、Value 是 attention 中的三组向量。
>
>Q 和 K 用来计算 attention weights，决定每个 token 应该关注哪些 token；V 是被加权汇总的信息内容。
>
>一句话理解：
>
>**Q/K 决定看谁，V 决定拿什么。**

## 🔗 Connections

- [[Self-Attention]]
- [[Causal Attention]]
- [[Causal Mask]]
- [[Cross-Attention]]
- [[Artificial Intelligence/Transformer/Multi-Head Attention]]
- [[Attention Head]]
- [[Rotary Position Embedding]]
- [[KV Cache]]
- [[Transformer Block]]