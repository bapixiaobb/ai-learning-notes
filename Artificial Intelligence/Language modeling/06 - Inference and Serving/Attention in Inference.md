#AI #LanguageModeling

这里我们只讨论 [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>) 中的 Inference，所以我们这里用的是 [self](<../../Transformer/Self-Attention.md>) [causal attention](<../../Transformer/Causal%20Attention.md>)

Attention 中的 [Query Key Value](<../../Transformer/Query%20Key%20Value.md>) 在不同场景下，sequence 长度会有不同，training 中完整的推导在 [Implementation of Multi-head Attention](<../../Transformer/Implementation%20of%20Multi-head%20Attention.md>) 这里我们讨论的是 inference 中的 $Q, K, V$

---
## Attention shape

先暂时不考虑 batch 和 heads。Attention 接收：

```math
Q\in\mathbb{R}^{T\times d}
```

```math
K,V\in\mathbb{R}^{S\times d}
```

这里：

- `T`：有多少个 query，也就是现在要给多少个 token positions 计算输出；
- `S`：有多少个 key/value，也就是每个 query 可以查看多少个 context positions，看多少上下文。

在 training 中如果是普通 full [self-attention](<../../Transformer/Self-Attention.md>) 那么
```math
\boxed{T=S=\text{seq}}
```
但是如果我们只需要最后一个位置的输出，那我们可以只计算这个位置的 query。

假设我们有 6 个位置，此时我们只需要：
```
Q for x₆
```
所以 query 数量是：$T=1$

但是 $x_6$ 要根据整段 context 计算：
```
[x₁,x₂,x₃,x₄,x₅,x₆]
```
因此它要面对 6 个 key/value positions：$S=6$

---
## [Prefill](<./Prefill.md>)：所有位置都发出 query

假设 prompt 长度是 5：
```
[x₁,x₂,x₃,x₄,x₅]
```
Prefill 要同时为这 5 个位置计算输出：
```
Q：5个位置都发出询问
K：5个位置都可以被匹配
V：5个位置都提供信息
```
所以这个时候
```math
T=S
```

---
## Incremental Inference

假设现在加入：$x_6$

我们只需要计算 $x_6$ 这个位置的新输出，所以只需要一个 query：$T=1$

但 $x_6$ 仍然需要从整段 context 中寻找信息：

```
[x₁,x₂,x₃,x₄,x₅,x₆]
```

所以它仍然需要面对整段 sequence 的 keys 和 values：$S=6$
