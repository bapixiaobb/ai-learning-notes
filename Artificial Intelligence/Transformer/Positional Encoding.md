#LanguageModeling #Transformer #PositionEncoding #Architecture

Positional Encoding 指把 token 的位置信息加入 Transformer 的方法。  
因为 [[Self-Attention]] 本身不天然知道 token 的顺序，所以 Transformer 需要额外机制表示 position information。

## 🧠 Core Idea

>[!note]
>Transformer 处理的是一组 token representations。
>
>如果没有 positional information，模型很难区分：
>
>```math
>[x_1, x_2, x_3]
>```
>
>和：
>
>```math
>[x_3, x_2, x_1]
>```
>
>因为 attention 本身主要根据 token representations 计算关系，而不自带顺序结构。

Positional Encoding 的作用是告诉模型：

- token 在 sequence 中的绝对位置；
- tokens 之间的相对距离；
- sequence 的顺序结构。

## 📍 Why Position Matters

语言是有顺序的。

例如：

```math
\text{dog bites man}
```

和：

```math
\text{man bites dog}
```

包含相同 tokens，但意思不同。

>[!important]
>Token identity 不够。
>
>Language model 还需要知道 token order。

这就是 Positional Encoding 必须存在的原因。

## 🧩 Main Types

常见 positional encoding 方法包括：

| Method                         | Main Idea                                                |
| ------------------------------ | -------------------------------------------------------- |
| Absolute Positional Embedding  | 给每个 absolute position 一个 embedding，并加到 token embedding 上 |
| Sinusoidal Positional Encoding | 用固定 sin / cos 函数表示 position                              |
| Relative Position Information  | 让模型感知 token 之间的相对距离                                      |
| [[Rotary Position Embedding]]  | 在 attention 中旋转 $Q,K$，让 attention score 感知相对位置           |
| ALiBi                          | 给 attention score 加上和距离有关的 bias                          |

>[!note]
>不同方法的区别主要在于：
>
>position information 是加到 token representation 上，还是进入 attention score 的计算过程。

## 🏗️ Where It Enters the Model

Positional information 可以在不同位置进入 Transformer。

### 1. Add to token embeddings

例如 absolute positional embedding：

```math
h_t
=
e_t
+
p_t
```

其中：

- $e_t$ 是 token embedding；
- $p_t$ 是 position embedding。

### 2. Enter attention computation

例如 [[Rotary Position Embedding]]：

```math
Q,K
\rightarrow
\mathrm{RoPE}(Q,K)
```

这类方法让 position information 影响 attention scores。

>[!note]
>现代 LLM 中常见的是第二类：position information 进入 attention computation，而不是简单加到 embedding 上。

## ⚖️ Absolute vs Relative Position

| Type | Focus |
|---|---|
| absolute position | token 在 sequence 中的具体位置，例如第 17 个 token |
| relative position | 两个 token 之间相隔多远，例如相隔 3 个 tokens |

在 language modeling 中，relative position 往往很重要。  
比如模型经常需要关注：

- 前一个 token；
- 前几个 tokens；
- 最近的 delimiter；
- 当前 token 和某个 earlier token 的距离。

>[!note]
>现代 LLM 更关注如何让 attention 机制有效感知 relative position information。

## 🦙 In Modern LLMs

在 [[Llama-style Architecture]] 中，常见 position encoding choice 是 [[Rotary Position Embedding]]。

RoPE 不直接把 position vector 加到 token embedding 上，而是作用在 attention 中的 $Q,K$ 上：

```math
q_t \rightarrow R_t q_t
```

```math
k_t \rightarrow R_t k_t
```

这样 attention score 可以感知 token positions 之间的相对关系。

## 🚫 Common Confusions

### Positional Encoding 不是 Token Embedding

[[Token Embedding]] 表示 token 是什么；Positional Encoding 表示 token 在哪里。

### Positional Encoding 不是 Causal Mask

Positional Encoding 告诉模型 token 的顺序位置；[[Causal Mask]] 限制模型不能看 future tokens。

### RoPE 是一种 Positional Encoding

[[Rotary Position Embedding]] 是 Positional Encoding 的一种具体方法，常见于 modern decoder-only LLM。

---

>[!summary] My Understanding
>Positional Encoding 的核心作用是让 Transformer 感知 token order。
>
>因为 Self-Attention 本身不自带顺序结构，所以需要额外的 position information。
>
>不同 positional encoding 方法的区别在于：它们如何表示位置，以及这些位置信息是在 embedding 层进入模型，还是在 attention computation 中影响 attention scores。

## 🔗 Connections

- [[Token Embedding]]
- [[Self-Attention]]
- [[Transformer]]
- [[Transformer Block]]
- [[Rotary Position Embedding]]
- [[Causal Mask]] 
- [[Llama-style Architecture]]