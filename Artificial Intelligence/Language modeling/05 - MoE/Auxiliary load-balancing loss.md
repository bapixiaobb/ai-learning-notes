#LanguageModeling #AI

这是一种用来解决 MoE imbalance 问题的方法，主要可以抵消 rich-get-richer 的问题
## Main idea

在 normal language modeling loss 之外，加一个额外的 控制 expert 使用均匀性的 auxiliary loss。
```math
\boxed{\text{total loss}=\text{language modeling loss} + \lambda\times\text{load balancing loss}}
```
这里的 $\lambda$ 是一个人为设置的权重，控制“希望负载均衡”的要求有多强

>**Warning**
>这个 auxiliary loss 不是从 [Next-token prediction](<../02%20-%20Language%20Modeling%20Basics/Next-token%20prediction.md>) 的理论目标中自然推出来的
>而是一个工程上非常有效的 heuristic: 它逼迫 [router](<MoE%20Router.md>) 不要把所有 tokens 都送到少数 experts。

接下来用这个 **total loss** 做 backward。Balancing loss 会给 router 提供额外的 gradient；optimizer 据此更新 router 参数，让之后的 routing 更倾向于均衡使用 experts。

## Switch-style

经典的 load balancing loss 大概有两个量：
- **实际分配比例 ($f_i$):** 每个 expert 实际收到多少 tokens，下标 $i$ 表示某个 expert
- **平均 routing probability ($P_i)$:** router 给每个 expert 分配多少 probability mass

Switch 的形式是：
```math
L_{\mathrm{bal}}=N\sum_i f_iP_i
```
这里 $N$ 是 expert 数量。我们有两个 experts

## Example

>**Example** — 假设有 10 个 tokens、两个 experts，router 给出的概率是：
>
>| Tokens | 给 A 的概率 | 给 B 的概率 | 实际选择 |
>| ------ | ------- | ------- | ---- |
>| 其中 9 个 | 0.6     | 0.4     | A    |
>| 另外 1 个 | 0.4     | 0.6     | B    |
>

- **实际分配比例 ($f_i$):** $$f_A=9/10=0.9,\qquad f_B=1/10=0.1$$
- **平均 routing probability ($P_i)$:**$$P_A=\frac{9\times 0.6+1\times 0.4}{10}=0.58,\qquad P_B=0.42$$

这里可以看出一个现象：**A 实际拿到 90% 的 tokens，但平均概率是 58%**。因为选择只看谁更大：`0.6 > 0.4`，这个 token 就全部交给 A。

因此：
```math
L_{\mathrm{bal}} =2(0.9\times0.58+0.1\times0.42) =\boxed{1.128}
```
再像前面那样，把它乘上权重 $\lambda$，加入总 loss。

#### 为什么反向传播能推动均衡？

关键在于：本次 backward 把 $f_A,f_B$ 当作固定的统计结果，通过概率 $P_A,P_B$ 求梯度。

本次 loss 就是：

```math
L_{\mathrm{bal}}=1.8P_A+0.2P_B
```

可以理解成：**给热门的 A 分配概率，代价更高；给冷门的 B 分配概率，代价较低。**

具体看，如果把 `0.01` 的平均概率从 A 移给 B：
- A 的概率下降，让 loss 减少 $1.8\times0.01=0.018$；
- B 的概率上升，让 loss 增加 $0.2\times0.01=0.002$；
- 合起来，**loss 减少了 `0.016`**。
