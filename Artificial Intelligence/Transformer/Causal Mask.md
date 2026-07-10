#LanguageModeling #Transformer #Attention #Architecture

Causal Mask 是 decoder-only language model 中用来阻止模型看到 future tokens 的 mask。  
它保证第 $t$ 个位置只能使用当前位置及之前的信息。

Causal Mask 是实现 [[Causal Attention]] 的 attention mask。  
它在 self-attention 的 score matrix 上屏蔽 future positions，使第 $t$ 个 token 只能 attend to $1,\dots,t$。

## 🧠 Core Idea

>[!note]
>Causal Mask 的核心作用是：
>
>让模型在预测下一个 token 时不能偷看未来。
>
>也就是第 $t$ 个 token position 只能 attend to：
>
>```math
>x_1, x_2, \dots, x_t
>```
>
>不能 attend to：
>
>```math
>x_{t+1}, x_{t+2}, \dots
>```

这和 [[Autoregressive Language Model]] 的建模目标一致：

```math
p(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T}
p(x_t \mid x_{<t})
```

## 🔍 Why It Is Needed

在训练 [[Decoder-Only Transformer]] 时，整个 sequence 通常会一次性输入模型：

```math
[x_1, x_2, x_3, x_4]
```

如果没有 mask，第 1 个位置可能直接看到 $x_2, x_3, x_4$。  
这样模型在预测 next token 时就会“作弊”。

>[!important]
>Causal Mask 保证训练时虽然可以并行处理整个 sequence，但每个位置的信息访问仍然满足 causal constraint。

也就是说：

```math
\text{parallel computation}
\neq
\text{seeing future tokens}
```

## 🧩 Mask Pattern

对于长度为 $T=4$ 的 sequence，causal mask 可以理解成下三角结构：

```math
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
```

其中：

- $1$ 表示可以 attend；
- $0$ 表示不能 attend。

第 3 行表示第 3 个 token 可以看：

```math
x_1, x_2, x_3
```

但不能看：

```math
x_4
```

## 🎭 In Self-Attention

普通 attention score 是：

```math
\frac{QK^\top}{\sqrt{d_k}}
```

加入 Causal Mask 后：

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
```

其中 $M$ 是 mask matrix。

通常做法是把 future positions 的 score 设为：

```math
-\infty
```

这样经过 [[Softmax]] 后，对应 attention weight 就会变成：

```math
0
```

>[!note]
>Causal Mask 不改变 $Q,K,V$ 的来源。
>
>它改变的是每个 token position 可以 attend to 哪些 positions。

## 🧵 Relation to Causal Attention

[[Causal Attention]] 指带有 causal constraint 的 self-attention。  
Causal Mask 是实现这个 constraint 的常见方式。

可以理解为：

```math
\text{Self-Attention}
+
\text{Causal Mask}
=
\text{Causal Attention}
```

>[!note]
>Self-attention 说明 token positions 可以互相看。
>
>Causal mask 限制哪些 positions 不能被看。

## 📍 Causal Mask vs Positional Encoding

Causal Mask 和 [[Positional Encoding]] 是两件不同的事。

| Concept | Role |
|---|---|
| [[Positional Encoding]] | 告诉模型 token 在哪里 |
| Causal Mask | 告诉模型哪些 token 不能看 |
| [[Rotary Position Embedding]] | 把 position information 注入 attention |
| [[Causal Attention]] | 只能看 prefix 的 attention pattern |

>[!important]
>RoPE 不能替代 Causal Mask。
>
>RoPE 让模型知道位置关系；Causal Mask 阻止模型看到 future tokens。

## 🚀 Training vs Inference

在 training 中，Causal Mask 允许模型一次性处理整个 sequence，同时保持 autoregressive constraint。

训练时可以并行计算：

```math
p(x_2 \mid x_1),\quad
p(x_3 \mid x_{\leq 2}),\quad
p(x_4 \mid x_{\leq 3})
```

在 inference 中，模型通常已经只有 prefix 作为输入：

```math
[x_1, x_2, \dots, x_t]
```

然后预测：

```math
x_{t+1}
```

>[!note]
>训练时 Causal Mask 很关键，因为 future tokens 已经在输入 sequence 里。
>
>推理时模型通常逐步生成，future tokens 还不存在，但实现中仍然会保留 causal attention 逻辑。

## 🚫 Common Confusions

### Causal Mask 不是 Positional Encoding

Causal Mask 不告诉模型“第几个位置”。  
它只限制 attention 的可见范围。

### Causal Mask 不是 Padding Mask

Padding mask 用来忽略 padding tokens。  
Causal mask 用来屏蔽 future tokens。

二者可以同时存在。

### Causal Mask 不等于 RoPE

[[Rotary Position Embedding]] 编码 position information。  
Causal Mask 控制 information flow direction。

### Causal Mask 不改变模型参数

Causal Mask 通常不是 learnable parameter。  
它是 attention computation 中固定或动态生成的 mask。

---

>[!summary] My Understanding
>Causal Mask 是 decoder-only language model 中保证 autoregressive constraint 的机制。
>
>它让第 $t$ 个 token position 只能 attend to prefix，不能看到 future tokens。
>
>这样模型在训练时可以并行处理整个 sequence，同时仍然只能基于过去信息预测下一个 token。
>
>Causal Mask 解决的是“能不能看未来”的问题；Positional Encoding 解决的是“知不知道位置”的问题。

## 🔗 Connections

- [[Causal Attention]]
- [[Self-Attention]]
- [[Decoder-Only Transformer]]
- [[Transformer Block]]
- [[Training vs Inference]]
- [[Positional Encoding]]
- [[Rotary Position Embedding]]