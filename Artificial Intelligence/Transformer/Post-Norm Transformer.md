#DeepLearning #NeuralNetwork #Transformer #Normalization #Architecture

[Post-Norm Transformer](<./Post-Norm%20Transformer.md>) 指把 normalization 放在 residual addition 之后的 Transformer block 结构。

核心形式是：

```math
x_{\text{out}}
=
\mathrm{Norm}(x + F(x))
```

其中：

- $x$ 是输入 hidden state / [Residual Stream](<./Residual%20Stream.md>)；
- $F$ 是 sublayer，例如 [Self-Attention](<./Self-Attention.md>) 或 [MLP](<./MLP.md>)；
- $x + F(x)$ 是 [Residual Connection](<./Residual%20Connection.md>)；
- $\mathrm{Norm}$ 可以是 [Layer Normalization](<./Layer%20Normalization.md>) 或其他 normalization 方法。

## 🧠 Core Idea

>**Note**
>Post-Norm 的意思是：
>
>先进入 sublayer，再 residual add，最后 normalization。
>
>也就是：
>
>```math
>\mathrm{SubLayer}
>\rightarrow
>\mathrm{Residual Add}
>\rightarrow
>\mathrm{Norm}
>```

这和 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) 相反。  
Pre-Norm 是先 normalization，再进入 sublayer：

```math
x_{\text{out}}
=
x + F(\mathrm{Norm}(x))
```

## 🧩 In Transformer Block

一个 Post-Norm [Transformer Block](<./Transformer%20Block.md>) 通常有两次 Post-Norm update。

### Attention update

```math
x'
=
\mathrm{Norm}
\left(
x + \mathrm{Attention}(x)
\right)
```

### MLP update

```math
x_{\text{out}}
=
\mathrm{Norm}
\left(
x' + \mathrm{MLP}(x')
\right)
```

合起来：

```math
x'
=
\mathrm{Norm}
\left(
x + \mathrm{Attention}(x)
\right)
```

```math
x_{\text{out}}
=
\mathrm{Norm}
\left(
x' + \mathrm{MLP}(x')
\right)
```

>**Note**
>Post-Norm 中，normalization 直接作用在 residual addition 之后的结果上。

## 🏛️ In Original Transformer

[Original Transformer](<./Original%20Transformer.md>) 使用的是 Post-Norm 结构。

经典形式可以写成：

```math
\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))
```

这里的 sublayer 可以是：

- [Multi-Head Attention](<./Multi-Head%20Attention.md>)
- Cross-Attention
- [Feed-Forward Network](<../Neural%20Networks/Feed-Forward%20Network.md>)

>**Note**
>Post-Norm 是原始 Transformer 论文中的常见 block layout。
>
>但 modern decoder-only LLM 更常使用 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>)。

## ⚖️ Post-Norm vs Pre-Norm

| Structure | Formula | Norm Position |
|---|---|---|
| [Post-Norm Transformer](<./Post-Norm%20Transformer.md>) | $\mathrm{Norm}(x + F(x))$ | after residual add |
| [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) | $x + F(\mathrm{Norm}(x))$ | before sublayer |

>**Important**
>Post-Norm 和 Pre-Norm 的区别不是用什么 normalization formula，而是 normalization 放在哪里。
>
>[Layer Normalization](<./Layer%20Normalization.md>) / [RMSNorm](<./RMSNorm.md>) 是 norm method；Post-Norm / Pre-Norm 是 norm placement。

## 🧵 Relation to Residual Stream

在 Post-Norm 中，residual connection 仍然存在：

```math
x + F(x)
```

但是 residual addition 之后立刻经过 normalization：

```math
\mathrm{Norm}(x + F(x))
```

这意味着主 hidden state 每经过一个 sublayer 后都会被重新 normalized。

>**Note**
>Post-Norm 中，normalization 位于 residual stream 的主路径上。
>
>相比之下，Pre-Norm 中 residual stream 的 identity path 更直接。

这也是为什么在很深的 Transformer 中，Pre-Norm 通常更容易训练。

## 🧠 Why Post-Norm Can Be Harder for Deep Models

Post-Norm 的 residual path 不是简单的：

```math
x \rightarrow x + F(x)
```

而是：

```math
x \rightarrow \mathrm{Norm}(x + F(x))
```

Normalization 会改变 residual addition 后的尺度和方向。

>**Note**
>这不代表 Post-Norm 错。
>
>它只是通常在 very deep Transformers 中优化更困难，因此 modern LLM 更常采用 Pre-Norm。

直觉上，Pre-Norm 给 gradient 提供了更直接的 residual path；Post-Norm 中这条 path 会经过 normalization。

## 🚫 Common Confusions

### 1. Post-Norm 不是 LayerNorm

[Layer Normalization](<./Layer%20Normalization.md>) 是一种 normalization 方法。  
[Post-Norm Transformer](<./Post-Norm%20Transformer.md>) 是一种 block layout。

Original Transformer 中常见的是：

```math
\mathrm{LayerNorm}(x + F(x))
```

这里同时出现了：

- LayerNorm：normalization formula；
- Post-Norm：normalization placement。

### 2. Post-Norm 不等于 encoder-decoder Transformer

[Original Transformer](<./Original%20Transformer.md>) 是 encoder-decoder structure，并且使用 Post-Norm。  
但 Post-Norm 本身只是 block 内部的 norm placement。

### 3. Post-Norm 不改变 attention mask

Post-Norm 只决定 norm 放在哪里。  
是否 causal 由 [Causal Mask](<./Causal%20Mask.md>) 或 attention pattern 决定。

## 🧾 My Understanding

>**Summary**
>[Post-Norm Transformer](<./Post-Norm%20Transformer.md>) 的核心公式是：
>
>```math
>x_{\text{out}} = \mathrm{Norm}(x + F(x))
>```
>
>它先做 sublayer transformation 和 residual addition，再做 normalization。
>
>[Original Transformer](<./Original%20Transformer.md>) 使用的是 Post-Norm，而 modern decoder-only LLM 更常用 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>)。
>
>理解 Post-Norm 的关键是：
>
>**Post-Norm / Pre-Norm 讨论的是 normalization placement，不是 normalization formula。**

## 🔗 Connections

- [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>)
- [Original Transformer](<./Original%20Transformer.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Normalization](<./Normalization.md>)
- [Layer Normalization](<./Layer%20Normalization.md>)
- [RMSNorm](<./RMSNorm.md>)
- [Residual Connection](<./Residual%20Connection.md>)
- [Residual Stream](<./Residual%20Stream.md>)
- [Self-Attention](<./Self-Attention.md>)
- [MLP](<./MLP.md>)
- [Feed-Forward Network](<../Neural%20Networks/Feed-Forward%20Network.md>)