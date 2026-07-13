#DeepLearning #NeuralNetwork #Transformer #Normalization #Architecture

[Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) 指把 normalization 放在 sublayer 之前的 Transformer block 结构。

核心形式是：

```math
x_{\text{out}}
=
x
+
F(\mathrm{Norm}(x))
```

其中：

- $x$ 是输入 hidden state / [Residual Stream](<./Residual%20Stream.md>)；
- $\mathrm{Norm}$ 可以是 [Layer Normalization](<./Layer%20Normalization.md>) 或 [RMSNorm](<./RMSNorm.md>)；
- $F$ 是某个 sublayer，例如 [Self-Attention](<./Self-Attention.md>) 或 [MLP](<./MLP.md>)；
- $x + F(\mathrm{Norm}(x))$ 是 [Residual Connection](<./Residual%20Connection.md>)。

## 🧠 Core Idea

>**Note**
>Pre-Norm 的意思是：
>
>先 normalization，再进入 sublayer，最后把 sublayer output 加回 residual stream。
>
>也就是：
>
>```math
>\mathrm{Norm}
>\rightarrow
>\mathrm{SubLayer}
>\rightarrow
>\mathrm{Residual Add}
>```

这和 [Post-Norm Transformer](<./Post-Norm%20Transformer.md>) 不同。  
Post-Norm 是先 residual add，再 normalization：

```math
x_{\text{out}}
=
\mathrm{Norm}(x + F(x))
```

![pre-norm transformer block.png](<./pre-norm%20transformer%20block.png>)
## 🧩 In Transformer Block

一个 Pre-Norm [Transformer Block](<./Transformer%20Block.md>) 通常有两次 Pre-Norm update：

### Attention update

```math
a
=
\mathrm{Attention}(\mathrm{Norm}(x))
```

```math
x'
=
x + a
```

### MLP update

```math
m
=
\mathrm{MLP}(\mathrm{Norm}(x'))
```

```math
x_{\text{out}}
=
x' + m
```

合起来：

```math
x'
=
x
+
\mathrm{Attention}(\mathrm{Norm}(x))
```

```math
x_{\text{out}}
=
x'
+
\mathrm{MLP}(\mathrm{Norm}(x'))
```

>**Note**
>Attention 和 MLP 都读取 normalized hidden states，但写回的是原来的 residual stream。

## 🧵 Relation to Residual Stream

在 Pre-Norm Transformer 中，[Residual Stream](<./Residual%20Stream.md>) 有一条比较直接的路径穿过所有 layers。

因为每个 sublayer 的输出是加到 $x$ 上：

```math
x_{\text{out}}
=
x
+
F(\mathrm{Norm}(x))
```

所以 residual stream 可以保持这种逐层累加的形式：

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
>Pre-Norm 的一个重要直觉是：
>
>normalization 发生在 branch 里面，而不是直接挡在 residual stream 的主路径上。

这使得 information 和 gradient 更容易沿着 residual path 传播。

## ⚖️ Pre-Norm vs Post-Norm

| Structure | Formula | Norm Position |
|---|---|---|
| [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) | $x + F(\mathrm{Norm}(x))$ | before sublayer |
| [Post-Norm Transformer](<./Post-Norm%20Transformer.md>) | $\mathrm{Norm}(x + F(x))$ | after residual add |

>**Important**
>[Original Transformer](<./Original%20Transformer.md>) 使用的是 Post-Norm。
>
>Modern decoder-only LLM 更常使用 Pre-Norm，常见搭配是 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) + [RMSNorm](<./RMSNorm.md>)。

## 🦙 In Llama-style Architecture

在 [Llama-style Architecture](<./Llama-style%20Architecture.md>) 中，Pre-Norm 通常和 [RMSNorm](<./RMSNorm.md>) 搭配。

一个 block 可以写成：

```math
x'
=
x
+
\mathrm{CausalAttention}(\mathrm{RMSNorm}(x))
```

```math
x_{\text{out}}
=
x'
+
\mathrm{SwiGLUMLP}(\mathrm{RMSNorm}(x'))
```

这里：

- [RMSNorm](<./RMSNorm.md>) 是 normalization method；
- Pre-Norm 是 normalization placement；
- [Residual Connection](<./Residual%20Connection.md>) 是加法结构；
- [Residual Stream](<./Residual%20Stream.md>) 是主 hidden state 流。

## 🧠 Why Pre-Norm Helps

Pre-Norm 常用于 deep Transformer，因为它通常更稳定。

直觉上：

- residual path 更直接；
- gradient 更容易穿过很多 layers；
- 每个 sublayer 接收到的输入尺度更稳定；
- 深层模型更容易优化。

如果写成：

```math
x_{\text{out}}
=
x
+
F(\mathrm{Norm}(x))
```

那么对 $x$ 的梯度中有一条 identity-like path：

```math
\frac{\partial x_{\text{out}}}{\partial x}
\approx
I
+
\frac{\partial F(\mathrm{Norm}(x))}{\partial x}
```

>**Note**
>这个 identity path 是 residual connection 带来的。
>
>Pre-Norm 让这条 path 更直接，因此在很深的 Transformer 中更常见。

## 🚫 Common Confusions

### 1. Pre-Norm 不是一种 normalization 方法

Pre-Norm 说的是 norm 的位置。  
具体 norm 可以是：

- [Layer Normalization](<./Layer%20Normalization.md>)
- [RMSNorm](<./RMSNorm.md>)

### 2. RMSNorm 不等于 Pre-Norm

[RMSNorm](<./RMSNorm.md>) 是 normalization formula。  
[Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) 是 block layout。

它们经常一起出现，但不是同一个概念。

### 3. Pre-Norm 不改变 causal mask

Pre-Norm 只改变 normalization placement。  
是否 causal 由 [Causal Attention](<./Causal%20Attention.md>) 和 [Causal Mask](<./Causal%20Mask.md>) 决定。

---

>**Summary** — My Understanding
>[Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) 的核心公式是：
>
>```math
>x_{\text{out}} = x + F(\mathrm{Norm}(x))
>```
>
>它把 normalization 放在 attention / MLP 之前，再把 sublayer output 加回 residual stream。
>
>在 modern decoder-only LLM 中，常见组合是 Pre-Norm + RMSNorm。Pre-Norm 强调 norm placement，RMSNorm 强调 norm formula。

## 🔗 Connections

- [Transformer Block](<./Transformer%20Block.md>)
- [Post-Norm Transformer](<./Post-Norm%20Transformer.md>)
- [Normalization](<./Normalization.md>)
- [Layer Normalization](<./Layer%20Normalization.md>)
- [RMSNorm](<./RMSNorm.md>)
- [Residual Connection](<./Residual%20Connection.md>)
- [Residual Stream](<./Residual%20Stream.md>)
- [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)
- [Llama-style Architecture](<./Llama-style%20Architecture.md>)
- [Self-Attention](<./Self-Attention.md>)
- [MLP](<./MLP.md>)