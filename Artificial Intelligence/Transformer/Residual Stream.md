#DeepLearning #NeuralNetwork #Transformer #Architecture #LanguageModeling

Residual Stream 指 Transformer 中贯穿所有 layers 的主 hidden state 流。  
它可以理解为每个 token position 当前拥有的 representation，而每个 [Transformer Block](<./Transformer%20Block.md>) 都在这个主 representation 上添加新的更新量。

## 🧠 Core Idea

>**Note**
>Residual Stream 是 Transformer 中 hidden states 的主数据流。
>
>每一层的 attention 和 MLP 并不是完全重写 token representation，而是向 residual stream 中加入新的信息。

如果第 $\ell$ 层的输入是：

```math
X^{(\ell)}
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

那么经过一个 sublayer 后，常见形式是：

```math
X^{(\ell+1)}
=
X^{(\ell)}
+
\Delta^{(\ell)}
```

其中：

- $X^{(\ell)}$ 是当前 residual stream；
- $\Delta^{(\ell)}$ 是 attention 或 MLP 计算出的更新量；
- $X^{(\ell+1)}$ 是更新后的 residual stream。

>**Important**
>Residual Stream 不是一个额外的 layer，也不是一个独立模块。
>
>它是多个 [Residual Connection](<./Residual%20Connection.md>) 串联后形成的主 hidden state 路径。

## 🧩 Why It Exists

Transformer 很深时，如果每一层都完全重写 representation，信息和 gradient 都很难稳定传递。

Residual stream 提供了一条相对直接的路径：

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

每一层只需要学习一个增量：

```math
\Delta^{(\ell)}
```

而不是重新生成整个 hidden representation。

>**Note**
>可以把 residual stream 理解成一个不断被编辑的“工作记忆”。
>
>每个 block 读取当前 hidden states，计算新的信息，然后把这些信息加回主流中。

## 🏗️ Residual Stream in Transformer Block

在 modern [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>) 中，一个 [Transformer Block](<./Transformer%20Block.md>) 通常有两次 residual update：

1. attention update；
2. MLP update。

以 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) 为例：

```math
A^{(\ell)}
=
\mathrm{Attention}
\left(
\mathrm{Norm}(X^{(\ell)})
\right)
```

```math
\tilde{X}^{(\ell)}
=
X^{(\ell)}
+
A^{(\ell)}
```

然后：

```math
M^{(\ell)}
=
\mathrm{MLP}
\left(
\mathrm{Norm}(\tilde{X}^{(\ell)})
\right)
```

```math
X^{(\ell+1)}
=
\tilde{X}^{(\ell)}
+
M^{(\ell)}
```

这里：

- $X^{(\ell)}$ 是 block 输入时的 residual stream；
- $A^{(\ell)}$ 是 attention branch 的更新；
- $\tilde{X}^{(\ell)}$ 是 attention 后的 residual stream；
- $M^{(\ell)}$ 是 MLP branch 的更新；
- $X^{(\ell+1)}$ 是 block 输出时的 residual stream。

## 🔁 Attention and MLP as Updates

在 residual stream 视角下，attention 和 MLP 都是在写入主 hidden state。

### Attention Update

[Self-Attention](<./Self-Attention.md>) 读取当前 residual stream 中所有 token positions 的信息，计算 token mixing 的结果：

```math
A^{(\ell)}
=
\mathrm{Attention}
\left(
\mathrm{Norm}(X^{(\ell)})
\right)
```

然后写回 residual stream：

```math
\tilde{X}^{(\ell)}
=
X^{(\ell)}
+
A^{(\ell)}
```

>**Note**
>Attention update 的作用是把 context information 写入每个 token representation。

### MLP Update

[MLP](<./MLP.md>) 读取 attention 更新后的 residual stream，对每个 token position 独立做 nonlinear transformation：

```math
M^{(\ell)}
=
\mathrm{MLP}
\left(
\mathrm{Norm}(\tilde{X}^{(\ell)})
\right)
```

然后写回 residual stream：

```math
X^{(\ell+1)}
=
\tilde{X}^{(\ell)}
+
M^{(\ell)}
```

>**Note**
>MLP update 的作用是把 per-token nonlinear features 写入 residual stream。
>
>它不负责 token positions 之间的信息交换；token mixing 主要由 attention 完成。

## 📐 Shape of Residual Stream

Residual stream 的 shape 通常是：

```math
X
\in
\mathbb{R}^{B \times T \times d_{\text{model}}}
```

其中：

- $B$ 是 batch size；
- $T$ 是 sequence length；
- $d_{\text{model}}$ 是 hidden dimension。

如果省略 batch dimension，也可以写成：

```math
X
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

Transformer block 的一个重要性质是输入输出 shape 通常保持不变：

```math
T \times d_{\text{model}}
\rightarrow
T \times d_{\text{model}}
```

>**Note**
>这个 shape invariant 让多个 Transformer blocks 可以直接堆叠。
>
>每个 block 都读取同样 shape 的 residual stream，并输出同样 shape 的 residual stream。

## 🧵 Token-wise View

Residual stream 可以从 token position 的角度理解。

对于第 $t$ 个 token，在第 $\ell$ 层的 hidden state 是：

```math
x_t^{(\ell)}
\in
\mathbb{R}^{d_{\text{model}}}
```

attention 和 MLP 会不断更新它：

```math
x_t^{(\ell+1)}
=
x_t^{(\ell)}
+
\Delta_t^{(\ell)}
```

其中：

- $x_t^{(\ell)}$ 是当前 token representation；
- $\Delta_t^{(\ell)}$ 是这一层写入的新信息；
- $x_t^{(\ell+1)}$ 是更新后的 token representation。

>**Note**
>随着 layers 加深，每个 token 的 residual stream 会逐渐编码更多上下文信息。
>
>在 decoder-only LM 中，位置 $t$ 的 representation 最终用于预测下一个 token。

## 🔍 Residual Stream vs Hidden State

在很多语境下，residual stream 和 hidden states 会被近似混用，但它们强调的重点不同。

| Term | Emphasis |
|---|---|
| hidden states | 某一层输出的 token representations |
| residual stream | 贯穿多个 layers、被 residual updates 不断修改的主数据流 |

>**Note**
>可以把每一层的 hidden states 看成 residual stream 在某个深度上的状态。
>
>Residual stream 强调的是“同一个主表示流被逐层更新”的视角。

## ⚖️ Residual Stream vs Residual Connection

Residual Stream 和 [Residual Connection](<./Residual%20Connection.md>) 相关，但不是同一个概念。

| Concept | Meaning |
|---|---|
| [Residual Connection](<./Residual%20Connection.md>) | 具体的加法结构：$x + F(x)$ |
| Residual Stream | 多个 residual connections 串起来后形成的主 hidden state 流 |

例如：

```math
x_{\text{out}}
=
x
+
F(x)
```

这里的加法是 [Residual Connection](<./Residual%20Connection.md>)。

而在整个 Transformer 中：

```math
X^{(0)}
\rightarrow
X^{(1)}
\rightarrow
\cdots
\rightarrow
X^{(L)}
```

这条不断被更新的路径就是 Residual Stream。

>**Important**
>Residual connection 是局部结构。
>
>Residual stream 是全局数据流视角。

## 🧮 Why Residual Stream Matters for Interpretability

在 Transformer interpretability 中，residual stream 是一个很重要的视角。

因为每个 block 都会向 residual stream 写入信息，所以可以把模型看成：

```math
\text{embedding}
+
\text{layer 1 updates}
+
\text{layer 2 updates}
+
\cdots
+
\text{layer L updates}
```

最终的 residual stream 再被 Language Modeling Head 映射到 logits。

>**Note**
>从这个角度看，Transformer 的每一层都在逐步编辑 token representation。
>
>最终 logits 不是某一层单独产生的，而是所有层对 residual stream 的累积更新结果。

这也解释了为什么 residual stream 常用于分析：

- information flow；
- attention head contribution；
- MLP contribution；
- logit lens；
- mechanistic interpretability。

## 🧠 Residual Stream in Decoder-Only LM

在 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>) 中，residual stream 有一个特别重要的约束：  
第 $t$ 个 token position 的 residual stream 不能包含 future tokens 的信息。

也就是说：

```math
x_t^{(\ell)}
\text{ can depend on }
x_{\leq t}^{(0)}
```

但不能依赖：

```math
x_{>t}^{(0)}
```

这个因果约束由 [Causal Attention](<./Causal%20Attention.md>) 和 [Causal Mask](<./Causal%20Mask.md>) 保证。

>**Note**
>因此，在 decoder-only LM 中，每个位置最终的 residual stream 都是 prefix representation。
>
>这个 representation 再经过 Language Modeling Head 预测 next token。

## 🔄 Forward Pass View

在 forward pass 中，residual stream 是主要被传递的 tensor。

粗略地说：

1. token ids 经过 [Token Embedding](<./Token%20Embedding.md>) 得到初始 residual stream；
2. 每个 [Transformer Block](<./Transformer%20Block.md>) 读取 residual stream；
3. attention branch 向 residual stream 写入 token-mixing information；
4. MLP branch 向 residual stream 写入 nonlinear feature information；
5. 最终 residual stream 经过 final norm 和 output head 变成 logits。

>**Note**
>Forward pass 不是只沿着 attention branch 走。
>
>Attention 和 MLP 都是在 residual stream 上做更新。

这和 [Forward Pass in Transformer](<./Forward%20Pass%20in%20Transformer.md>) 直接相关。

## 🚫 Common Confusions

### 1. Residual Stream 不是 residual error

>**Warning**
>Residual stream 里的 residual 不是 regression 里的 residual error。
>
>这里的 residual 指的是 residual connection，也就是保留原输入并加上一个 learned update。

回归里的 residual 通常是：

```math
y - \hat{y}
```

而 Transformer 里的 residual connection 是：

```math
x + F(x)
```

这两个概念不应该混在一起。  
更详细的区别见 [Residual Connection vs Regression Residual](<./Residual%20Connection%20vs%20Regression%20Residual.md>)。

### 2. Residual Stream 不是 attention output

Attention output 只是写入 residual stream 的其中一种更新量。

```math
\text{residual stream}
\neq
\text{attention output}
```

更准确地说：

```math
\text{updated residual stream}
=
\text{old residual stream}
+
\text{attention output}
```

### 3. Residual Stream 不是额外参数

Residual stream 是 forward pass 中流动的 activations，不是模型参数。

模型参数在：

- attention projection matrices；
- MLP weights；
- normalization parameters；
- embedding matrix；
- output head。

Residual stream 是这些参数作用在输入 token sequence 上产生的 hidden activations。

---

>**Summary** — My Understanding
>Residual Stream 是 Transformer 中贯穿所有 blocks 的主 hidden state 流。
>
>每个 block 都会读取当前 residual stream，然后通过 attention 和 MLP 计算更新量，再用 [Residual Connection](<./Residual%20Connection.md>) 把这些更新量加回去。
>
>因此，Transformer 不是每层完全重写 representation，而是不断在已有 representation 上添加信息。
>
>理解 residual stream 的关键是：
>
>**[Residual Connection](<./Residual%20Connection.md>) 是局部的加法结构，Residual Stream 是这些加法结构串联起来后形成的全局信息流。**

## 🔗 Connections

- [Transformer](<./Transformer.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)
- [Forward Pass in Transformer](<./Forward%20Pass%20in%20Transformer.md>)
- [Residual Connection](<./Residual%20Connection.md>)
- [Residual Connection vs Regression Residual](<./Residual%20Connection%20vs%20Regression%20Residual.md>)
- [Self-Attention](<./Self-Attention.md>)
- [MLP](<./MLP.md>)
- [Normalization](<./Normalization.md>)