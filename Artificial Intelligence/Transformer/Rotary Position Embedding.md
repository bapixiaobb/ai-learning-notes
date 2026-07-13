#LanguageModeling #Transformer #PositionEncoding #Architecture

Rotary Position Embedding, 简称 RoPE, 是一种把 position information 注入 [Self-Attention](<./Self-Attention.md>) 的方法。它常见于 modern decoder-only LLM, 例如 [Llama-style Architecture](<./Llama-style%20Architecture.md>)。

>**Note**
> RoPE 不是把 position embedding 加到 token embedding 上。
>
> RoPE 是在 attention 里, 根据 token position 去**旋转**每个 token 的 query 和 key vector。

## 一句话直觉

attention 的核心是 query 和 key 做点积 $q\cdot k$。我们希望这个点积反映的是两个 token 的**相对距离** (差几个位置), 而不是它们各自的绝对位置。比如 "第 5 个词关注第 3 个词" 和 "第 105 个词关注第 103 个词" 应该是同一种关系。

RoPE 的做法: 把位置 $m$ 的 query 旋转 $m\theta$, 位置 $n$ 的 key 旋转 $n\theta$。旋转后做点积, 两个绝对角度互相抵消, 只剩下 $(m-n)\theta$。

>**Important**
> RoPE 把 absolute position 通过旋转变成 attention score 里的 relative position effect。

## Before RoPE: Token 到 QKV

>**Note**
> 一个 token 在模型里先是一个 integer token id。经过 [Token Embedding](<./Token%20Embedding.md>) 后, 每个 token 会变成一个 vector。

假设一个 sequence 有 $T$ 个 tokens, 每个 token 的 hidden dimension 是 $d_{model}$:

```math
X \in \mathbb{R}^{T \times d_{model}}
```

可以把它理解成:

```text
row    = token / position
column = hidden dimension / feature
```

在 attention 中, 每个 token vector 会通过三个 linear projections 生成 [Query Key Value](<./Query%20Key%20Value.md>):

```math
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
```

>**Note**
> Q 和 K 决定 attention score, 也就是“谁关注谁”。
>
> V 是被 attention weights 加权汇总的信息内容。

RoPE 作用在投影**之后**的 $Q$ 和 $K$ 上, 通常不作用在 $V$ 上。

>**Warning**
> 投影 ($XW$) 和旋转是两件分开的事。
>
> 投影是普通 linear layer, 先把 token representation 变成 query/key。RoPE 旋转的是投影后的 query/key vector, 它不去乘 learnable weight。

## Core Idea

>**Important**
> RoPE 对每个 token 的 query/key vector, 在 hidden dimension 内部**两两分组**, 然后根据这个 token 的 position 把每一组当成一个二维向量去**旋转**。

关键: RoPE **不拆 token 那一维 (rows)**, 它拆的是**每一行内部的 columns**。

```text
q_t = [a, b, c, d]   --- 某个 token 的 query vector

(a, b) = pair 0      --- 当成一个 2D 向量旋转
(c, d) = pair 1      --- 当成另一个 2D 向量旋转
```

不同 token 用各自的 position 决定转多少; 同一 token 内不同的 pair 用不同的 frequency 决定转多快。

输出 shape 不变:

```math
Q_{rope}, K_{rope} \in \mathbb{R}^{T \times d_k}
```

## 二维旋转公式

对一个二维向量旋转角度 $\alpha$:

```math
\begin{bmatrix} a' \\ b' \end{bmatrix}
=
\begin{bmatrix} \cos\alpha & -\sin\alpha \\ \sin\alpha & \cos\alpha \end{bmatrix}
\begin{bmatrix} a \\ b \end{bmatrix}
```

展开:

```math
a' = a\cos\alpha - b\sin\alpha, \qquad b' = a\sin\alpha + b\cos\alpha
```

旋转有两个重要性质:

- 不改变向量模长
- $R(\alpha)^\top R(\beta) = R(\beta-\alpha)$

## 为什么旋转能编码相对位置

只看一对维度。位置 $m$ 的 query 这一对记为 $q$, 位置 $n$ 的 key 这一对记为 $k$。RoPE 把它们分别旋转 $m\theta$ 和 $n\theta$:

```math
q_{rope} = R(m\theta)q, \qquad k_{rope} = R(n\theta)k
```

attention score 用点积:

```math
q_{rope}^\top k_{rope}
= \big(R(m\theta)q\big)^\top \big(R(n\theta)k\big)
= q^\top R(m\theta)^\top R(n\theta)k
= q^\top R\big((n-m)\theta\big)k
```

>**Important**
> 结果只依赖 $(n-m)$, 也就是两个 token 的位置差。
>
> 这就是为什么 RoPE 可以让 attention score 感知 relative position information。

## Angle 从哪里来: Position × Frequency

旋转角度由两件事决定:

```math
\text{angle}_{i} = \text{position} \times \omega_i
```

- **position**: token 在 sequence 里的位置, 决定转多少。
- **$\omega_i$**: 第 $i$ 对维度的 frequency, 决定这一对转多快。

不同 pair 使用不同 frequency:

```math
\omega_i = \theta^{-2i/d_k}, \qquad i = 0, 1, \dots, \tfrac{d_k}{2}-1
```

其中 $\theta$ 是超参, 常取 10000。

```text
i 小: omega 接近 1 -> 转得快 -> 高频 -> 对近距离变化敏感
i 大: omega 很小    -> 转得慢 -> 低频 -> 捕捉更大范围的位置差
```

## 手算例子

设 $d_k=4$, 位置 $m=1$, query 是:

```math
q=[x_0, x_1, x_2, x_3]
```

分成:

```text
pair0 = (x0, x1)
pair1 = (x2, x3)
```

频率:

```math
\omega_0 = 10000^0 = 1, \qquad \omega_1 = 10000^{-2/4}=0.01
```

角度:

```math
\text{angle}_0 = 1, \qquad \text{angle}_1 = 0.01
```

各自旋转:

```math
\begin{aligned}
x_0' &= x_0\cos 1 - x_1\sin 1 & x_2' &= x_2\cos 0.01 - x_3\sin 0.01\\
x_1' &= x_0\sin 1 + x_1\cos 1 & x_3' &= x_2\sin 0.01 + x_3\cos 0.01
\end{aligned}
```

>**Note**
> 高频 pair 转得多, 低频 pair 转得少。
>
> 输入和输出 shape 都还是 $1 \times 4$。

## 分块对角矩阵视角

把所有 pair 拼起来, RoPE 等价于乘一个分块对角矩阵。每个 $2\times2$ block 只作用在一对 hidden dimensions 上。

```math
\begin{bmatrix} x_0' \\ x_1' \\ x_2' \\ x_3' \end{bmatrix}
=
\begin{bmatrix}
\cos\alpha_0 & -\sin\alpha_0 & 0 & 0\\
\sin\alpha_0 & \cos\alpha_0 & 0 & 0\\
0 & 0 & \cos\alpha_1 & -\sin\alpha_1\\
0 & 0 & \sin\alpha_1 & \cos\alpha_1
\end{bmatrix}
\begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix}
```

>**Note**
> 实际 implementation 不会真的造这个稀疏大矩阵。
>
> 代码里会直接按 pair 做旋转。具体实现见 [RoPE Implementation](<./RoPE%20Implementation.md>)。

## Where RoPE Appears

attention 分支里:

```math
X \rightarrow Q,K,V
```

然后:

```math
Q\rightarrow\mathrm{RoPE}(Q), \qquad K\rightarrow\mathrm{RoPE}(K)
```

再算:

```math
\mathrm{Attention}(Q_{rope}, K_{rope}, V)
```

>**Note**
> RoPE 影响 attention score, 因为 score 来自 $QK^\top$。
>
> 它通常不作用在 $V$, 因为 $V$ 是被取走的信息内容。

## 和 Causal Mask 的区别

| Concept | Role |
|---|---|
| RoPE | 注入 position information, 尤其是 relative position effect |
| [Causal Mask](<./Causal%20Mask.md>) | 防止看到 future tokens |
| [Causal Attention](<./Causal%20Attention.md>) | 只能 attend to prefix 的 attention pattern |

>**Note**
> RoPE 告诉模型 token 的位置关系。
>
> Causal mask 告诉模型哪些位置不能看。

## My Current Understanding

>**Summary**
> RoPE 是一种 positional encoding, 不把 position vector 加到 embedding 上, 而是在 attention 里旋转 $Q$ 和 $K$。
>
> 对每个 token, 在它自己的 query/key vector 内沿 hidden dimension 两两分组做二维旋转, 角度 = position × frequency。
>
> 核心原理是: 旋转后做点积, $q^\top R((n-m)\theta)k$ 只依赖位置差 $(n-m)$, 所以 absolute position 被翻译成了 attention score 里的 relative position effect。

## Related

- [RoPE Implementation](<./RoPE%20Implementation.md>)
- [Positional Encoding](<./Positional%20Encoding.md>)
- [Self-Attention](<./Self-Attention.md>)
- [Query Key Value](<./Query%20Key%20Value.md>)
- [Causal Attention](<./Causal%20Attention.md>)
- [Causal Mask](<./Causal%20Mask.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)
- [Llama-style Architecture](<./Llama-style%20Architecture.md>)
