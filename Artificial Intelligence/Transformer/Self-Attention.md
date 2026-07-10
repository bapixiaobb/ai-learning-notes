#LanguageModeling #Transformer #Attention #Architecture

Self-Attention 是 Transformer 中让同一个 sequence 内部的 tokens 互相交换信息的机制。  
它让每个 token 根据其他 tokens 的信息更新自己的 representation。

## 🧠 Core Idea

>[!note]
>Self-Attention 的核心是：
>
>每个 token 都可以根据同一个 sequence 中的其他 tokens 来更新自己。
>
>也就是说，token representation 不再只表示这个 token 本身，而是变成 context-dependent representation。

例如：

```math
\text{river bank}
```

和：

```math
\text{bank account}
```

这里的 `bank` 是同一个词，但上下文不同。  
Self-Attention 可以让 `bank` 关注周围 tokens，从而得到不同的 representation。

## 🧩 Why “Self”?

“Self” 指的是 $Q,K,V$ 都来自同一个 sequence。

普通 attention 可以是：

```math
\text{query from A, key/value from B}
```

而 Self-Attention 是：

```math
\text{query, key, value all from the same sequence}
```

>[!note]
>Self-Attention 是 sequence 内部的信息交互。
>
>如果是 decoder attend to encoder outputs，那是 Cross-Attention。

## 🎭 Token Mixing

Self-Attention 的主要作用是 token mixing。

也就是让不同 token positions 之间交换信息：

```math
x_1, x_2, \dots, x_T
\rightarrow
h_1, h_2, \dots, h_T
```

其中每个 $h_t$ 都可以包含其他 tokens 的上下文信息。

>[!important]
>Self-Attention 负责 token positions 之间的信息交互。
>
>[[MLP]] 通常负责每个 token position 内部的 nonlinear transformation。

## 🔁 Relation to Causal Attention

Self-Attention 本身不一定是 causal 的。

如果每个 token 可以看整个 sequence，就是 bidirectional self-attention。  
如果加上 [[Causal Mask]]，只能看 prefix，就变成 [[Causal Attention]]。

```math
\text{Self-Attention}
+
\text{Causal Mask}
=
\text{Causal Attention}
```

---

>[!summary] My Understanding
>Self-Attention 是 Transformer 的核心机制。
>
>它让同一个 sequence 中的 token positions 互相交换信息，从而把 token embedding 更新成 context-dependent representation。
>
>在 decoder-only language model 中，Self-Attention 通常会加上 Causal Mask，变成 Causal Attention。

## 🔗 Connections

- [[Transformer]]
- [[Transformer Block]]
- [[Causal Attention]]
- [[Causal Mask]]
- [[MLP]]
- [[Query Key Value]] 
- [[Artificial Intelligence/Transformer/Multi-Head Attention]]