
## Formula

和 [softmax causal attention](<Self-Causal%20Attention.md>) 一样的 [Q, K, V](<Query%20Key%20Value.md>):
```math
\boxed{o_t=\sum_{i\le t}v_i(k_i^\top q_t)= \left(\sum_{i\le t}v_i k_i^\top\right)q_t}
```
**这里 $q_t、k_i、v_i$ 都是列向量，$o_t$ 也是列向量**。所以 $k_i^\top q_t$ 是一个数，用它给整个 $v_i$ 加权。

- 不使用 [softmax](<Softmax.md>)：直接把内积 $k_i^\top q_t$ 当权重，没有指数，也没有除以权重总和。
- 保留 [causal](<Causal%20Mask.md>) 限制：只加到 $i=t$，当前 token 只能使用自己和之前的 key/value。

## Causal

因为 attention 是 causal 的，不同位置能使用的范围不同：

| 当前输出位置  | 使用哪个 query | 允许使用哪些 key/value |
| ------- | ---------- | ---------------- |
| 第 1 行输出 | $Q_{1,:}$  | 第 1 行            |
| 第 2 行输出 | $Q_{2,:}$  | 前 2 行            |
| 第 3 行输出 | $Q_{3,:}$  | 前 3 行            |
| 第 4 行输出 | $Q_{4,:}$  | 前 4 行            |
在线性注意力里，可以把这些前缀分别需要的中间结果写成：
```math
\begin{aligned} \text{前1行的累计结果}&=K_{1,:}^\top V_{1,:}\\ \text{前2行的累计结果}&=\text{前1行的结果}+K_{2,:}^\top V_{2,:}\\ \text{前3行的累计结果}&=\text{前2行的结果}+K_{3,:}^\top V_{3,:} \end{aligned}
```
这是**同一个训练样本内部，各个位置之间的计算关系**。它可以写成递推，不代表训练实现必须逐 token 串行执行。
