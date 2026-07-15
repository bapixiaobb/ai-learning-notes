#DeepLearning #NeuralNetwork #Transformer #Architecture

Residual Connection 是 neural network 中的一种加法连接结构。
它把某一层的输入直接加到该层学到的 transformation output 上。

基本形式是：

```math
x_{\text{out}}
=
x
+
F(x)
```

其中：

- $x$ 是输入 representation；
- $F(x)$ 是 neural network sublayer 学到的 transformation；
- $x_{\text{out}}$ 是更新后的 representation。

## 🧠 Core Idea

>**Note**
>Residual connection 的核心思想是：
>
>一层网络不需要完全重新生成一个新的 representation，而是学习如何在原 representation 上做增量更新。

也就是：

```math
\text{new representation}
=
\text{old representation}
+
\text{learned update}
```

在这个视角下：

```math
F(x)
=
x_{\text{out}}
-
x
```

所以 $F(x)$ 可以理解为从旧 representation 到新 representation 的 update / correction。

## 🧩 Why It Matters

深层 neural networks 很难训练，其中一个原因是信息和 gradients 需要穿过很多 layers。

如果每一层都完全重写 representation：

```math
x
\rightarrow
F(x)
```

那么原始信息可能在深层传播中逐渐丢失。

Residual connection 提供了一条更直接的信息路径：

```math
x
\rightarrow
x + F(x)
```

>**Note**
>Residual connection 让 layer 可以选择“少改一点”。
>
>如果某一层暂时学不到有用 transformation，它可以让 $F(x)$ 接近 0，于是：
>
>
```math
>x_{\text{out}} \approx x
>
```
>
>这让深层网络更容易优化。

## 🔁 Residual Connection in Transformer

在 [Transformer Block](<./Transformer%20Block.md>) 中，[Self-Attention](<./Self-Attention.md>) 和 [MLP](<./MLP.md>) 通常都会通过 residual connection 写回主 hidden states。

以 modern Pre-Norm Transformer 为例：

```math
x'
=
x
+
\mathrm{Attention}(\mathrm{Norm}(x))
```

然后：

```math
x_{\text{out}}
=
x'
+
\mathrm{MLP}(\mathrm{Norm}(x'))
```

这里有两个 residual connections：

1. attention output 加回原 hidden state；
2. MLP output 加回 attention 更新后的 hidden state。

>**Note**
>在 Transformer 中，residual connection 是 [Residual Stream](<./Residual%20Stream.md>) 形成的局部结构。
>
>多个 residual connections 串联起来，就形成了贯穿所有 blocks 的主 hidden state 流。

## 🧱 Residual Connection vs Plain Layer

没有 residual connection 时，一层网络通常是：

```math
x_{\text{out}}
=
F(x)
```

有 residual connection 时：

```math
x_{\text{out}}
=
x
+
F(x)
```

两者的区别是：

| Structure | Formula | Meaning |
|---|---|---|
| plain layer | $F(x)$ | 直接用 transformation output 替换输入 |
| residual layer | $x + F(x)$ | 保留输入，并加上 learned update |

>**Important**
>Residual connection 改变的是 [computation graph](<../Neural%20Networks/Computational%20Graph.md>)。
>
>它不是 loss function，也不是 prediction error。

Regression 中的 residual 是：

```math
y - \hat{y}
```

这和 residual connection 不是同一个概念，更详细的区别见 [Residual Connection vs Regression Residual](<./Residual%20Connection%20vs%20Regression%20Residual.md>)。

## 📐 Shape Requirement

Residual connection 要求 $x$ 和 $F(x)$ 的 shape 可以相加。

最常见情况是：

```math
x \in \mathbb{R}^{T \times d_{\text{model}}}
```

```math
F(x) \in \mathbb{R}^{T \times d_{\text{model}}}
```

因此：

```math
x + F(x)
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

>**Note**
>这也是为什么 [Transformer Block](<./Transformer%20Block.md>) 的输入和输出通常保持同样 shape。
>
>只有 shape 一致，residual addition 才能直接发生。

如果 sublayer 改变了维度，通常需要使用 projection 让 shape 对齐。

## 🧮 Gradient Flow

Residual connection 对 gradient flow 很重要。

如果：

```math
y
=
x
+
F(x)
```

那么：

```math
\frac{\partial y}{\partial x}
=
I
+
\frac{\partial F(x)}{\partial x}
```

这里的 $I$ 表示 identity path。

>**Note**
>这个 identity path 给 gradient 提供了一条更直接的传播路径。
>
>即使 $\frac{\partial F(x)}{\partial x}$ 很小，gradient 仍然可以通过 identity component 向前传播。

这就是 residual connection 能帮助训练 deep networks 的一个重要原因。

## 🧵 Relation to Residual Stream

Residual Connection 是局部加法结构：

```math
x_{\text{out}}
=
x
+
F(x)
```

[Residual Stream](<./Residual%20Stream.md>) 是这些局部 residual connections 在多层 Transformer 中串联形成的主数据流：

```math
X^{(0)}
\rightarrow
X^{(1)}
\rightarrow
\cdots
\rightarrow
X^{(L)}
```

>**Note**
>Residual connection 是“某一层如何把输入加回去”。
>
>Residual stream 是“整个 Transformer 里 hidden states 如何沿着这些加法结构一路流动”。

这两个概念相关，但不应该混在一起。

## 🧠 Intuition in Transformer

在 Transformer 中，可以把每个 block 想成对 residual stream 做两次编辑：

```math
\text{old hidden state}
+
\text{attention update}
+
\text{MLP update}
```

也就是：

```math
x_{\text{out}}
=
x
+
\Delta_{\text{attn}}
+
\Delta_{\text{mlp}}
```

其中：

- $\Delta_{\text{attn}}$ 写入 contextual / token mixing information；
- $\Delta_{\text{mlp}}$ 写入 per-token nonlinear features。

>**Note**
>Transformer block 不是把 $x$ 丢掉再重新算一个 representation。
>
>它是在 $x$ 的基础上不断写入新的信息。

## 🏗️ Post-Norm and Pre-Norm

Residual connection 和 normalization 的相对位置形成了两类常见 Transformer block。

### Post-Norm

[Original Transformer](<./Original%20Transformer.md>) 使用 Post-Norm：

```math
x_{\text{out}}
=
\mathrm{Norm}(x + F(x))
```

也就是先 residual addition，再 normalization。

### Pre-Norm

Modern LLM 更常用 Pre-Norm：

```math
x_{\text{out}}
=
x
+
F(\mathrm{Norm}(x))
```

也就是先 normalization，再进入 sublayer，最后加回 residual stream。

>**Note**
>Pre-Norm 的 residual path 更直接，因此通常更适合训练很深的 Transformer。
>
>这也是 modern [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>) 常用 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) 的原因之一。

## 🚫 Common Confusions

### 1. Residual connection 不是 regression residual

Regression residual 是：

```math
r = y - \hat{y}
```

Residual connection 是：

```math
x_{\text{out}} = x + F(x)
```

它们不是同一个概念。

### 2. Residual connection 不是 skip connection 的全部

Residual connection 可以看作一种 skip connection，但不是所有 skip connection 都一定是 residual addition。

有些 skip connection 可能是 concatenation 或其他连接方式。

### 3. Residual connection 不是一个单独参数

Residual connection 本身通常没有参数。
真正有参数的是 $F(x)$ 里面的 layers，例如 attention projection matrices 或 MLP weights。

---

>**Summary** — My Understanding
>Residual Connection 是 neural network 中的加法结构：
>
>
```math
>x_{\text{out}} = x + F(x)
>
```
>
>它让 layer 学习对输入 representation 的增量更新，而不是完全替换输入。
>
>在 Transformer 中，attention 和 MLP 都通过 residual connection 把更新量写回主 hidden states。多个 residual connections 串联起来形成 [Residual Stream](<./Residual%20Stream.md>)。
>
>理解 residual connection 的关键是：
>
>**它是网络内部的信息流结构，不是 regression 中的 prediction residual。**

## 🔗 Connections

- [Residual Stream](<./Residual%20Stream.md>)
- [Residual Connection vs Regression Residual](<./Residual%20Connection%20vs%20Regression%20Residual.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Transformer](<./Transformer.md>)
- [Self-Attention](<./Self-Attention.md>)
- [MLP](<./MLP.md>)
- [Normalization](<./Normalization.md>)
