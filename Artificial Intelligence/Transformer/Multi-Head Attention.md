#LanguageModeling #Transformer #Attention #Architecture

Multi-Head Attention 是把 attention 分成多个 Attention Head 并行计算的机制。  
每个 head 都有自己的 Q、K、V projections，可以从不同 representation subspace 学习不同的 token interaction pattern。实现上通常用一个大的 projection matrix 一次性算出所有 heads 的 Q/K/V，然后 reshape/split 出 head dimension。详见 [[Implementation of Multi-head Attention]] 

## 🧠 Core Idea

>[!note]
>普通 attention 可以看作只用一个 head 做信息汇总。
>
>Multi-Head Attention 则是让多个 heads 同时做 attention，然后把结果拼接起来。

直觉上：

$$
\text{one attention head}
\rightarrow
\text{one way of looking at context}
$$

$$
\text{multiple attention heads}
\rightarrow
\text{multiple ways of looking at context}
$$

不同 heads 可能关注不同类型的信息，例如：

- local context；
- long-distance dependency；
- syntax relation；
- repeated pattern；
- delimiter / structure token。

## 🧩 Basic Formula

给定 input hidden states：

$$
X \in \mathbb{R}^{T \times d_{\text{model}}}
$$

第 $i$ 个 head 会计算自己的：

$$
Q_i = XW_i^Q
$$

$$
K_i = XW_i^K
$$

$$
V_i = XW_i^V
$$

然后做 attention：

$$
\mathrm{head}_i
=
\mathrm{Attention}(Q_i, K_i, V_i)
$$

多个 heads 的输出会被 concat：

$$
\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)
$$

最后再经过一个 output projection：

$$
\mathrm{MHA}(X)
=
\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W_O
$$

## 📐 Shape View

通常有：

$$
d_{\text{model}}
=
h \cdot d_{\text{head}}
$$

其中：

- $h$ 是 number of heads；
- $d_{\text{head}}$ 是每个 head 的 dimension。

如果：

$$
X \in \mathbb{R}^{T \times d_{\text{model}}}
$$

那么每个 head 里：

$$
Q_i, K_i, V_i
\in
\mathbb{R}^{T \times d_{\text{head}}}
$$

每个 head 输出：

$$
\mathrm{head}_i
\in
\mathbb{R}^{T \times d_{\text{head}}}
$$

拼接后：

$$
\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)
\in
\mathbb{R}^{T \times d_{\text{model}}}
$$

>[!note]
>Multi-head 的关键是：每个 head 在较小的 $d_{\text{head}}$ 子空间里做 attention，最后再合并回 $d_{\text{model}}$。

## 🎭 Why Multiple Heads?

如果只有一个 attention head，模型只有一种 attention pattern。  
多个 heads 允许模型同时学习多种 relation。

例如：

| Head Type | Possible Pattern |
|---|---|
| local head | 关注附近 tokens |
| syntax head | 关注语法相关 tokens |
| induction head | 关注重复模式 |
| delimiter head | 关注换行、括号、特殊符号 |
| long-range head | 关注远距离依赖 |

>[!note]
>这些只是直觉解释，不是每个 head 都一定有清晰的人类可解释功能。
>
>但 multi-head structure 确实增加了 attention 的表达能力。

## 🧵 Relation to Self-Attention

Multi-Head Attention 通常是在 [[Self-Attention]] 上做多头扩展。

可以理解为：

$$
\text{Self-Attention}
\rightarrow
\text{Multi-Head Self-Attention}
$$

如果再加上 [[Causal Mask]]，就得到 causal multi-head attention：

$$
\text{Multi-Head Self-Attention}
+
\text{Causal Mask}
=
\text{Causal Multi-Head Attention}
$$

在 [[Decoder-Only Transformer]] 中，Multi-Head Attention 通常是 causal 的。

## 🦙 Relation to Modern LLMs

[[Original Transformer]] 使用 standard Multi-Head Attention。

Modern LLM 中也常用 multi-head structure，但可能会修改 K/V 的组织方式：

| Variant                 | Main Idea                   |
| ----------------------- | --------------------------- |
| Multi-Head Attention    | 每个 query head 有自己的 K/V head |
| Multi-Query Attention   | 多个 query heads 共享一组 K/V     |
| Grouped Query Attention | 多个 query heads 分组共享 K/V     |

>[!note]
>这些 variants 的主要动机之一是减少 inference 时的 [[KV Cache]] memory cost。
>
>所以 attention head 的设计既是 [[Model Architecture]] 问题，也会影响 [[Systems for Language Models]]。

## 🚫 Common Confusions

### Multi-head 不是多个 Transformer blocks

多个 heads 是同一个 attention layer 内部的并行 attention subspaces。  
多个 blocks 是模型深度上的堆叠。

### Head 不是单独一个完整模型

每个 head 只负责 attention 中的一部分计算。  
多个 heads 的输出还需要 concat 和 output projection。

### Multi-head 不等于多个 tokens

Head 是 hidden dimension 的分解方式，不是 sequence length 的分解方式。

---

>[!summary] My Understanding
>Multi-Head Attention 是把 attention 拆成多个 heads 并行计算。
>
>每个 head 有自己的 Q、K、V projections，可以学习不同的 token interaction pattern。
>
>多个 heads 的结果会拼接后再投影回 $d_{\text{model}}$。
>
>在 modern LLM 中，standard multi-head attention 还可以进一步变成 [[Multi-Query Attention]] 或 [[Grouped Query Attention]]，以减少 KV cache cost。

![[TransformerLM.jpeg]]
## 🔗 Connections

- [[Self-Attention]]
- [[Query Key Value]]
- [[Attention Head]]
- [[Causal Attention]]
- [[Causal Mask]]
- [[Decoder-Only Transformer]]
- [[Transformer Block]]
- [[Multi-Query Attention]]
- [[Grouped Query Attention]]
- [[KV Cache]]
- [[Model Architecture]]
