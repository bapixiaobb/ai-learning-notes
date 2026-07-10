#LanguageModeling #Transformer #Attention #Architecture

Causal Attention 是一种带有 causal constraint 的 attention。  
它要求第 $t$ 个 token position 只能 attend to 自己和之前的 tokens，不能看 future tokens。

## 🧠 Core Idea

>[!note]
>Causal Attention 的核心是：
>
>```math
>x_t
>\text{ can attend to }
>x_1, x_2, \dots, x_t
>```
>
>但不能 attend to：
>
>```math
>x_{t+1}, x_{t+2}, \dots
>```

这使模型在预测下一个 token 时不能偷看答案。

## 🧩 Relation to Self-Attention

Causal Attention 通常是在 [[Self-Attention]] 上加 [[Causal Mask]] 得到的：

```math
\text{Self-Attention}
+
\text{Causal Mask}
=
\text{Causal Attention}
```

普通 self-attention 允许每个 token 看整个 sequence；  
causal attention 只允许每个 token 看 prefix。

| Attention Type | 第 $t$ 个 token 可以看什么 |
|---|---|
| [[Self-Attention]] | 通常可以看所有 positions |
| Causal Attention | 只能看 $1,\dots,t$ |

## 🎯 Why It Matters

Causal Attention 和 [[Autoregressive Language Model]] 的目标匹配。

Autoregressive language model 建模：

```math
p(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T}
p(x_t \mid x_{<t})
```

也就是说，预测当前 token 时只能依赖过去 tokens。

>[!important]
>Causal Attention 保证了 information flow 的方向：
>
>```math
>\text{past} \rightarrow \text{future}
>```
>
>而不是 future tokens 反过来影响 past positions。

## 🔍 Intuition

如果模型要预测：

```math
x_4
```

它只能使用：

```math
x_1, x_2, x_3
```

如果它能看到：

```math
x_4, x_5, \dots
```

训练就会变成作弊。

>[!note]
>Causal Attention 不是告诉模型 token 在哪里；它只是限制模型能看哪些位置。
>
>位置在哪里由 [[Positional Encoding]] / [[Rotary Position Embedding]] 提供。

## 🚫 Common Confusions

### Causal Attention 不是 Causal Mask

[[Causal Mask]] 是实现 causal constraint 的 mask。  
Causal Attention 是加了这种约束后的 attention pattern。

### Causal Attention 不是 Positional Encoding

Causal Attention 控制可见范围；  
[[Positional Encoding]] 表示位置。

---

>[!summary] My Understanding
>Causal Attention 是只能看 prefix 的 attention。
>
>它通常由 Self-Attention 加上 Causal Mask 实现，是 [[Decoder-Only Transformer]] 和 Next Token Prediction 匹配的关键机制。

---
## 🔗 Connections

- [[Self-Attention]]
- [[Causal Mask]]
- [[Decoder-Only Transformer]]
- [[Positional Encoding]]