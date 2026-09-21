#LanguageModeling #Transformer #Attention #Architecture

[QKV](<Query%20Key%20Value.md>) 仍然照常计算：

```math
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
```

[Decoder-Only Transformer](<../00%20-%20Maps%20and%20Architectures/Decoder-Only%20Transformer.md>) 一般都用这个，长这样：

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+\text{mask}
\right)V
```

我们现在讨论的 Attention 就是这个，mask 一般用 [Causal Mask](<Causal%20Mask.md>)，就是：
```math
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
```

通常做法是把 future positions 的 score 设为：$-\infty$，这样经过 [Softmax](<Softmax.md>) 后，对应 attention weight 就会变成：$0$

## Computation cost

#### Optimization with [GPU](<../../GPU%20and%20NPU/GPU.md>)：

[Flash Attention](<Flash%20Attention.md>) 可以把 $QK^\top \rightarrow \mathrm{softmax}\rightarrow AV$ 这整个 attention 过程做成一个 memory- efficient fused kernel，以解决 [GPU Memory Bound](<../../GPU%20and%20NPU/GPU%20Memory%20Bound.md>) 的问题。

保留标准 attention 的数学定义，优化计算与数据搬运
#### Mathematical Computation

先做一个**代数实验**：暂时去掉 softmax 和缩放，也先不加 causal mask：
```math
 \widetilde O=(QK^\top)V
```
这个式子用结合律：
```math
 \boxed{(QK^\top)V=Q(K^\top V)}
```
两种顺序的区别非常具体：

| 计算顺序 | 中间矩阵 | 主要计算量 |
| -------------------- | --------------- | ------------------ |
| 先算 $QK^\top$，再乘 $V$ | $T\times T$ | $O(T^2d_k+T^2d_v)$ |
| 先算 $K^\top V$，再乘 $Q$ | $d_k\times d_v$ | $O(Td_kd_v)$ |

但是：
```math
\operatorname{softmax}(QK^\top)V \;\neq\; Q(K^\top V)
```
结合律只能重排矩阵乘法。**去掉 softmax 后的点积加权，可以廉价地计算；它还不是原来那个 attention。**

由此，引入问题：**能不能重新设计权重，让它仍然表示 query 与 key 的匹配程度，同时允许这种计算重排？**

[Linear Attention](<Linear%20Attention.md>)：可以改变 attention 的数学形式，让计算量对序列长度线形增长
GDN：Linear Attention 的一种
