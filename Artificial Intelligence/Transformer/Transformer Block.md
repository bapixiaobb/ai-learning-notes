#DeepLearning #NeuralNetwork #Transformer #Architecture #LanguageModeling

[[Transformer Block]] 是 Transformer 中反复堆叠的基本 computation unit。  
在 language model 中，多个 Transformer blocks 会不断更新每个 token position 的 hidden representation。

## 🧠 Core Idea

>[!note]
>一个 [[Transformer Block]] 的核心作用是：
>
>在保持 sequence length 不变的情况下，更新每个 token 的 representation。
>
>它通常包含两类主要 transformation：
>
>- [[Self-Attention]]：让 token positions 之间交换信息；
>- [[MLP]] / [[Feed-Forward Network]]：对每个 token representation 做 nonlinear processing。

如果输入 hidden states 是：

```math
X \in \mathbb{R}^{T \times d_{\text{model}}}
```

那么一个 Transformer block 通常输出：

```math
X' \in \mathbb{R}^{T \times d_{\text{model}}}
```

也就是说，block 不改变 sequence length，也通常不改变 hidden dimension；它改变的是每个 token representation 的内容。

## 🧩 Where It Sits in Transformer

一个 Transformer 通常由多个 Transformer blocks 堆叠：

```math
X^{(0)}
\rightarrow
X^{(1)}
\rightarrow
X^{(2)}
\rightarrow
\cdots
\rightarrow
X^{(L)}
```

其中：

- $X^{(0)}$ 来自 [[Token Embedding]] 和 [[Positional Encoding]]；
- $X^{(\ell)}$ 是第 $\ell$ 层 block 输出的 hidden states；
- $L$ 是 number of layers。

在 [[Decoder-Only Transformer]] 中，所有 blocks 都使用 [[Causal Attention]]，保证每个位置不能看到 future tokens。

## 🏗️ Basic Components

一个 Transformer block 通常包含：

| Component | Role |
|---|---|
| [[Self-Attention]] | token positions 之间的信息交互 |
| [[MLP]] / [[Feed-Forward Network]] | 每个 token position 内部的 nonlinear transformation |
| [[Residual Connection]] | 保留主信息流，并让 block 学习增量更新 |
| [[Normalization]] | 稳定 hidden states 的尺度和 gradient flow |

>[!note]
>可以粗略理解为：
>
>[[Self-Attention]] 负责 token mixing；[[MLP]] 负责 feature processing；[[Residual Connection]] 负责信息主路径；[[Normalization]] 负责训练稳定性。

## 🔁 Data Flow in a Block

现代 decoder-only LLM 中常见的是 [[Pre-Norm Transformer]] 结构。

一个简化的 Pre-Norm Transformer block 可以写成：

```math
A
=
\mathrm{Attention}(\mathrm{Norm}(X))
```

```math
X'
=
X + A
```

```math
M
=
\mathrm{MLP}(\mathrm{Norm}(X'))
```

```math
X_{\text{out}}
=
X' + M
```

也就是：

```math
X
\rightarrow
\mathrm{Norm}
\rightarrow
\mathrm{Attention}
\rightarrow
+
\rightarrow
\mathrm{Norm}
\rightarrow
\mathrm{MLP}
\rightarrow
+
\rightarrow
X_{\text{out}}
```

>[!note]
>这里的两个加号就是 [[Residual Connection]]。
>
>它们让 attention 和 MLP 都作为对主 hidden state 的增量更新，而不是完全替换原来的 representation。

这条被不断更新的主 hidden state 路径通常可以理解成 [[Residual Stream]]。

## 🎭 Attention Sub-layer

Attention sub-layer 负责 token positions 之间的信息交互。

输入：

```math
X \in \mathbb{R}^{T \times d_{\text{model}}}
```

会先生成：

```math
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
```

然后通过 attention 计算每个 token position 应该从其他 positions 收集多少信息。

在 decoder-only language model 中，attention 是 causal 的：

```math
h_t
\text{ can attend to }
h_{\leq t}
```

不能 attend to：

```math
h_{>t}
```

>[!note]
>Attention sub-layer 的输出仍然是一个 shape 为 $T \times d_{\text{model}}$ 的 tensor。
>
>它更新的是每个 token representation 中关于上下文的信息。

具体的 $Q,K,V$、attention score 和 [[Softmax]] 机制放在 [[Self-Attention]] 和 [[Artificial Intelligence/Transformer/Multi-Head Attention]] 中整理。

## 🧮 MLP Sub-layer

MLP sub-layer 对每个 token position 独立作用。

如果忽略 batch dimension，一个 position 的 hidden vector 是：

```math
x_t \in \mathbb{R}^{d_{\text{model}}}
```

MLP 会做类似：

```math
\mathrm{MLP}(x_t)
=
W_2 \sigma(W_1 x_t)
```

其中：

- $W_1$ 通常把维度从 $d_{\text{model}}$ 扩大到 $d_{\text{ff}}$；
- $\sigma$ 是 activation function；
- $W_2$ 再把维度映射回 $d_{\text{model}}$。

所以：

```math
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
```

>[!note]
>MLP 不负责 token positions 之间的信息交换。
>
>它对每个位置独立作用，负责对已经混合了上下文的信息做 nonlinear transformation。

这也是为什么可以说：

> [[Self-Attention]] 是 token mixing；[[MLP]] 是 per-token feature transformation。

## 🧵 Residual Stream

在 Transformer block 中，主 hidden state 会通过 residual connections 一路贯穿所有 blocks。

可以粗略写成：

```math
X^{(\ell+1)}
=
X^{(\ell)}
+
\Delta^{(\ell)}
```

其中 $\Delta^{(\ell)}$ 来自 attention 或 MLP sub-layer。

>[!note]
>[[Residual Stream]] 可以理解为贯穿整个 Transformer 的主数据流。
>
>每个 block 不是从零生成新的 representation，而是在已有 representation 上添加更新量。

这和 [[Residual Connection]] 相关，但不是同一个概念：  
[[Residual Connection]] 是具体的加法结构，[[Residual Stream]] 是这种加法结构贯穿多层后形成的主信息流。

## ⚖️ Pre-Norm vs Post-Norm

Transformer block 中 normalization 的位置有两种常见形式。

### Post-Norm

[[Original Transformer]] 使用的是 Post-Norm：

```math
X'
=
\mathrm{Norm}(X + \mathrm{SubLayer}(X))
```

也就是先做 sublayer 和 residual addition，再做 normalization。

### Pre-Norm

现代 LLM 更常见的是 Pre-Norm：

```math
X'
=
X + \mathrm{SubLayer}(\mathrm{Norm}(X))
```

也就是先 normalization，再进入 sublayer，最后加回 residual stream。

>[!note]
>Pre-Norm 通常更适合训练很深的 Transformer，因为它让 gradient 更容易沿着 residual path 传播。
>
>这也是 modern decoder-only LLM 常用 [[Pre-Norm Transformer]] 的原因之一。

## 🦙 Modern LLM Block

在 [[Llama-style Architecture]] 中，一个 Transformer block 通常包含这些 modern choices：

| Part | Common Choice |
|---|---|
| attention | [[Causal Attention]] |
| position information | [[Rotary Position Embedding]] applied in attention |
| normalization | [[RMSNorm]] |
| norm placement | [[Pre-Norm Transformer]] |
| MLP activation | [[SwiGLU]] |
| attention variant | often [[Grouped Query Attention]] |

可以粗略写成：

```math
X'
=
X
+
\mathrm{Attention}(\mathrm{RMSNorm}(X))
```

```math
X_{\text{out}}
=
X'
+
\mathrm{SwiGLU\text{-}MLP}(\mathrm{RMSNorm}(X'))
```

>[!note]
>这不是 [[Original Transformer]] 的原始 block，而是 modern decoder-only LLM 中常见的 block pattern。

## 🔄 Forward Pass Through a Block

在 forward pass 中，Transformer block 会依次计算：

1. 对输入 hidden states 做 normalization；
2. 计算 attention output；
3. 把 attention output 加回 residual stream；
4. 再做 normalization；
5. 计算 MLP output；
6. 把 MLP output 加回 residual stream。

>[!important]
>Forward pass 不是只“走 attention”。
>
>一个完整 Transformer block 的 forward pass 通常包括 attention branch 和 MLP branch，并且每个 branch 都通过 residual connection 更新主 hidden states。

这可以连接到 [[Forward Pass in Transformer]]。

## 📐 Shape Invariants

Transformer block 中一个重要特点是 shape 通常保持不变：

```math
T \times d_{\text{model}}
\rightarrow
T \times d_{\text{model}}
```

虽然内部会有临时维度变化，例如：

```math
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
```

或者 multi-head attention 中：

```math
d_{\text{model}}
\rightarrow
h \times d_{\text{head}}
```

但 block 的输入和输出 shape 通常一致。

>[!note]
>这个 shape invariant 很重要，因为它允许多个 Transformer blocks 直接堆叠。
>
>如果每层输出 shape 都变了，后续 block 就不能直接接上。

## 🧠 Why Transformer Block Matters

Transformer block 是连接 architecture 和 computation 的核心单位。

它同时决定：

- token 如何交换信息；
- hidden states 如何 nonlinear update；
- residual stream 如何传播；
- normalization 如何稳定训练；
- 每一层的 FLOPs 和 activation memory；
- 整个模型如何 scale up 到更多 layers。

>[!note]
>当说一个模型有 $L$ layers 时，通常指它堆叠了 $L$ 个 Transformer blocks。
>
>但 block 内部采用什么 attention、norm、MLP、activation，仍然属于重要的 [[Model Architecture]] choices。

---

>[!summary] My Understanding
>[[Transformer Block]] 是 Transformer 中反复堆叠的基本结构单元。
>
>它接收一组 token hidden states，经过 attention、MLP、normalization 和 residual connection 后，输出 shape 相同但内容更新后的 hidden states。
>
>在 modern decoder-only LLM 中，一个 block 通常采用 Pre-Norm 结构：先 norm，再进入 attention 或 MLP，最后把结果加回 residual stream。
>
>理解 Transformer block 的关键是区分：
>
>- [[Self-Attention]] 负责 token mixing；
>- [[MLP]] 负责 per-token nonlinear transformation；
>- [[Residual Connection]] 是加法结构；
>- [[Residual Stream]] 是贯穿所有 blocks 的主 hidden state 流。

![[TransformerLM.jpeg]]
## 🔗 Connections

- [[Transformer]]
- [[Transformer Family]]
- [[Original Transformer]]
- [[Decoder-Only Transformer]]
- [[Llama-style Architecture]]
- [[Model Architecture]]
- [[Self-Attention]]
- [[Causal Attention]]
- [[Artificial Intelligence/Transformer/Multi-Head Attention]]
- [[MLP]]
- [[Feed-Forward Network]]
- [[Residual Stream]]
- [[Residual Connection vs Regression Residual]]
- [[Normalization]]
- [[Layer Normalization]]
- [[Training vs Inference]]
- [[Resource Accounting]]