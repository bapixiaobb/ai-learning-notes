#DeepLearning #NeuralNetwork #NLP #Transformer #Architecture

[[Original Transformer]] 指 Vaswani et al. 2017 提出的 encoder-decoder Transformer architecture，也就是经典论文 *Attention Is All You Need* 里的那张 Transformer 图。

它最初主要用于 Machine Translation 这类 sequence-to-sequence tasks，而不是现在最常见的 decoder-only LLM。

## 🧠 Core Idea

>[!note]
>[[Original Transformer]] 的核心思想是：
>
>不用 recurrence，也不用 convolution，而是完全基于 attention mechanism 来处理 sequence data。
>
>它通过 [[Self-Attention]] 建模序列内部 token 之间的关系，并通过 encoder-decoder structure 实现从 source sequence 到 target sequence 的生成。

原始 Transformer 的整体形式是：

```math
\text{source sequence}
\rightarrow
\text{encoder}
\rightarrow
\text{encoder hidden states}
\rightarrow
\text{decoder}
\rightarrow
\text{target sequence}
```

例如 machine translation 中：

```math
\text{English sentence}
\rightarrow
\text{encoder}
\rightarrow
\text{decoder}
\rightarrow
\text{French sentence}
```

## 🧭 Why It Matters

>[!note]
>理解 [[Original Transformer]] 很重要，因为后来的很多 Transformer variants 都可以看作从它的结构中删减、重组或修改而来。
>
>例如：
>
>- [[Encoder-Only Transformer]] 基本保留 encoder side；
>- [[Decoder-Only Transformer]] 基本保留 decoder side 的 causal self-attention 主线；
>- [[Encoder-Decoder Transformer]] 继续使用完整的 encoder-decoder structure；
>- [[Llama-style Architecture]] 是 modern decoder-only Transformer 的一种代表性结构组合。

所以，当看到 “Transformer” 这个词时，需要区分它是在指：

- 原始 encoder-decoder Transformer；
- Transformer 这一族 architecture；
- 现代 LLM 中的 decoder-only Transformer。

## 🏗️ Overall Architecture

[[Original Transformer]] 由两个主要部分组成：

| Part | Role |
|---|---|
| encoder | 读取 source sequence，生成 contextual representations |
| decoder | 根据 target prefix 和 encoder outputs 逐步生成 target sequence |

整体数据流可以写成：

```math
X_{\text{src}}
\rightarrow
\text{Encoder}
\rightarrow
H_{\text{enc}}
```

```math
X_{\text{tgt}, <t}, H_{\text{enc}}
\rightarrow
\text{Decoder}
\rightarrow
p(y_t \mid y_{<t}, X_{\text{src}})
```

其中：

- $X_{\text{src}}$ 是 source sequence；
- $X_{\text{tgt}, <t}$ 是 target prefix；
- $H_{\text{enc}}$ 是 encoder 输出的 source representations；
- decoder 在第 $t$ 步预测下一个 target token。

## 📥 Encoder

Encoder 负责处理 source sequence。

给定 source tokens：

```math
x_1, x_2, \dots, x_S
```

encoder 会输出每个 source position 的 contextual representation：

```math
H_{\text{enc}}
=
(h_1, h_2, \dots, h_S)
```

其中：

```math
h_i \in \mathbb{R}^{d_{\text{model}}}
```

### Encoder Block

Original Transformer 的 encoder 由多个 identical encoder layers 堆叠而成。

每个 encoder block 包含：

1. [[Multi-Head Self-Attention]]
2. [[Feed-Forward Network]]
3. [[Residual Connection]]
4. [[Layer Normalization]]

粗略结构：

```math
X
\rightarrow
\text{Self-Attention}
\rightarrow
\text{FFN}
\rightarrow
X'
```

>[!note]
>Encoder 中的 self-attention 通常是 bidirectional 的。
>
>也就是说，每个 source token 可以 attend to source sequence 中的所有 tokens。

形式上：

```math
h_i
\text{ can attend to }
h_1, h_2, \dots, h_S
```

这和 [[Decoder-Only Transformer]] 中的 [[Causal Attention]] 不同。

## 📤 Decoder

Decoder 负责生成 target sequence。

给定已经生成的 target prefix：

```math
y_1, y_2, \dots, y_{t-1}
```

decoder 预测：

```math
p(y_t \mid y_{<t}, X_{\text{src}})
```

也就是说，decoder 同时依赖：

1. 已经生成的 target prefix；
2. encoder 对 source sequence 的 representations。

### Decoder Block

Original Transformer 的 decoder block 比 encoder block 多一个 attention sublayer。

每个 decoder block 包含：

1. [[Masked Self-Attention]]
2. [[Cross-Attention]]
3. [[Feed-Forward Network]]
4. [[Residual Connection]]
5. [[Layer Normalization]]

粗略结构：

```math
Y
\rightarrow
\text{Masked Self-Attention}
\rightarrow
\text{Cross-Attention}
\rightarrow
\text{FFN}
\rightarrow
Y'
```

## 🎭 Three Kinds of Attention

Original Transformer 中最容易混的地方是：它不只有一种 attention。

### 1. Encoder Self-Attention

Encoder self-attention 发生在 source sequence 内部。

```math
\text{source tokens attend to source tokens}
```

它通常是 bidirectional 的：

```math
x_i
\text{ can attend to }
x_1, x_2, \dots, x_S
```

作用是让每个 source token 得到上下文化表示。

### 2. Decoder Masked Self-Attention

Decoder masked self-attention 发生在 target sequence 内部。

```math
\text{target tokens attend to previous target tokens}
```

为了 autoregressive generation，decoder 不能看到未来 target tokens：

```math
y_t
\text{ can attend to }
y_{\leq t}
```

不能看到：

```math
y_{>t}
```

这需要 [[Causal Mask]]。

>[!note]
>Masked self-attention 是 decoder 能进行 autoregressive generation 的关键。
>
>这部分机制后来也成为 [[Decoder-Only Transformer]] 的核心。

### 3. Encoder-Decoder Cross-Attention

Cross-attention 发生在 decoder 和 encoder outputs 之间。

```math
\text{target tokens attend to source representations}
```

在 cross-attention 中：

- queries 来自 decoder hidden states；
- keys 和 values 来自 encoder hidden states。

可以写成：

```math
Q = H_{\text{dec}} W_Q
```

```math
K = H_{\text{enc}} W_K
```

```math
V = H_{\text{enc}} W_V
```

然后：

```math
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
```

>[!note]
>Cross-attention 让 decoder 在生成 target token 时能够读取 source sentence 的信息。
>
>这就是 encoder-decoder Transformer 适合 translation / summarization 等 conditional generation tasks 的原因。

## 🧩 Input and Output Flow

Original Transformer 的输入输出可以理解成两个序列：

| Sequence | Meaning |
|---|---|
| source sequence | 输入序列，例如待翻译的句子 |
| target sequence | 输出序列，例如翻译后的句子 |

训练时，target sequence 通常右移一位作为 decoder input：

```math
\text{decoder input}
=
[y_0, y_1, \dots, y_{T-1}]
```

decoder target 是：

```math
[y_1, y_2, \dots, y_T]
```

也就是说，decoder 在每个位置学习预测下一个 target token。

>[!note]
>训练时可以并行计算所有 target positions 的 loss，因为 causal mask 会阻止每个位置看到未来 tokens。
>
>推理时通常是 autoregressive generation，需要一步一步生成 target tokens。

这也自然连接到 [[Training vs Inference]]。

## 🧱 Positional Encoding

Original Transformer 不使用 recurrence，所以模型本身不知道 token 顺序。

因此需要加入 [[Positional Encoding]]。

原始论文中使用的是 sinusoidal positional encoding：

```math
\text{token embedding}
+
\text{positional encoding}
```

>[!note]
>Position information 是 Transformer architecture 的必要组成部分。
>
>如果没有 positional encoding，self-attention 本身无法区分 token 的顺序。

现代 LLM 中更常见的是 [[Rotary Position Embedding]]，但这不是 Original Transformer 的原始设计。

## 🔁 Residual Connection and LayerNorm

Original Transformer 的每个 sublayer 外面都有 residual connection 和 layer normalization。

形式上可以写成：

```math
\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
```

这对应 [[Post-Norm Transformer]]。

>[!note]
>Original Transformer 使用的是 Post-Norm 结构。
>
>现代 LLM 中更常见的是 [[Pre-Norm Transformer]]，例如先 norm 再进入 attention / MLP。

Residual connection 的作用是让信息和 gradient 更容易穿过深层网络，但 [[Residual Stream]] 这个概念在 modern Transformer 解释中会更突出。

## 🧮 Feed-Forward Network

每个 encoder block 和 decoder block 中都有 position-wise feed-forward network。

对每个 position 独立作用：

```math
\mathrm{FFN}(x)
=
W_2 \sigma(W_1 x + b_1) + b_2
```

其中 $\sigma$ 在 Original Transformer 中通常是 ReLU。

>[!note]
>Self-attention 负责 token positions 之间的信息交互。
>
>FFN / MLP 负责对每个 token representation 做 nonlinear transformation。

现代 LLM 中常见的 [[SwiGLU]] 不是 Original Transformer 的原始配置，而是后来的 architecture choice。

## ⚖️ Original Transformer vs Decoder-Only Transformer

Original Transformer 和 modern decoder-only LLM 最重要的区别是结构不同。

| Aspect | Original Transformer | Decoder-Only Transformer |
|---|---|---|
| overall structure | encoder-decoder | decoder-only |
| encoder | yes | no |
| cross-attention | yes | usually no |
| decoder self-attention | masked / causal | causal |
| main use | sequence-to-sequence | autoregressive language modeling |
| typical examples | translation Transformer, T5-style models | GPT, LLaMA, Mistral |

>[!important]
>Modern LLM 通常不是完整的 Original Transformer。
>
>GPT / LLaMA-style models 一般没有 encoder，也没有 encoder-decoder cross-attention，而是使用 decoder-only causal self-attention。

## 🧭 Relation to Transformer Family

[[Original Transformer]] 在 [[Transformer Family]] 中处于源头位置。

可以粗略理解成：

```math
\text{Original Transformer}
\rightarrow
\begin{cases}
\text{Encoder-Only Transformer} \\
\text{Decoder-Only Transformer} \\
\text{Encoder-Decoder Transformer}
\end{cases}
```

其中：

- [[Encoder-Only Transformer]] 强调 bidirectional representation learning；
- [[Decoder-Only Transformer]] 强调 autoregressive generation；
- [[Encoder-Decoder Transformer]] 强调 conditional generation。

---

>[!summary] My Understanding
>[[Original Transformer]] 是最初的 encoder-decoder Transformer。
>
>它包含 encoder 和 decoder 两部分：encoder 用 bidirectional self-attention 处理 source sequence，decoder 用 masked self-attention 处理 target prefix，并通过 cross-attention 读取 encoder outputs。
>
>理解 Original Transformer 的关键是区分三种 attention：
>
>- encoder self-attention；
>- decoder masked self-attention；
>- encoder-decoder cross-attention。
>
>现代 GPT / LLaMA-style LLM 通常不是完整的 Original Transformer，而是从 Transformer family 中发展出的 [[Decoder-Only Transformer]]。

## 🔗 Connections

- [[Transformer]]
- [[Transformer Family]]
- [[Encoder-Decoder Transformer]]
- [[Encoder-Only Transformer]]
- [[Decoder-Only Transformer]]
- [[Llama-style Architecture]]
- [[Transformer Block]]
- [[Self-Attention]]
- [[Masked Self-Attention]]
- [[Causal Attention]]
- [[Causal Mask]]
- [[Cross-Attention]]
- [[Artificial Intelligence/Transformer/Multi-Head Attention]]
- [[Feed-Forward Network]]
- [[Residual Connection]]
- [[Residual Stream]]
- [[Layer Normalization]]
- [[Post-Norm Transformer]]
- [[Positional Encoding]]
- [[Machine Translation]]
- [[Sequence-to-Sequence]]
- [[Training vs Inference]]