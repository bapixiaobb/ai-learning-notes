#AI #LanguageModeling

DeepSeek V3 使用的一种 [MoE](<Mixture%20of%20Experts.md>) 的均衡方法

## Main idea

它的直觉是：给每个 expert 的 routing score 加一个可调 bias
- 如果某个 expert 被用的太多，就降低它的bias
- 如果某个 expert 被用的太少，就提高它的bias

这样 router 在选择 experts 时，就会受到这个 bias 影响，从而动态调节 expert 使用频率。

## Method

对于 expert $i$，定义：
- $s_i$：router 为当前 token 算出的原始 score。
- $b_i$：这个 expert 的 balancing bias，同一 expert 的 bias 对所有 tokens 共用。

选择 expert 时，比较：
```math
\boxed{\text{selection score}_i=s_i+b_i}
```
初始时所有 $b_i=0$，因此选择结果和原来一样。
## Example

>**Example** — 假设有 10 个 tokens、两个 experts，每个 token 选一个的例子：
>
>| Tokens | 原始 score：A | 原始 score：B | 实际选择 |
>| ------ | ------- | ------- | ---- |
>| 其中 8 个 | 0.6     | 0.4     | A   |
>| 其中 1 个 | 0.51    | 0.49   | A   |
>| 另外 1 个 | 0.4     | 0.6     | B    |

这里先把原始 scores 当作已经算好的数字。

这次 forward 后统计发现：**A 收到 9 个 tokens，B 收到 1 个。** 相对于平均每个 expert 收到 5 个，A 过载，B 负载不足。

**于是，在这个 training step 结束时调整 bias：**

- A 过载：降低 $b_A$。
- B 负载不足：提高 $b_B$。

设每次调整幅度为 $\gamma$。为了看清效果，这个例子取 $\gamma=0.02$：
```math
b_A=0-0.02=-0.02,\qquad b_B=0+0.02=+0.02
```

#### 为什么这样能改变分配？

为了单独观察 bias 的作用，假设下一次输入和原始 scores 暂时相同。看那个原本接近平局的 token：
```math
\begin{aligned} \text{A 的 selection score}&=0.51-0.02=0.49\\ \text{B 的 selection score}&=0.49+0.02=0.51 \end{aligned}
```
它现在就会选 **B**。而那 8 个明显偏向 A 的 tokens，调整后仍然是 `0.58 > 0.42`，继续选 A。因此，这个例子的分配从 **9∶1 变成了 8∶2**。

每个 training step 都可以重新统计负载，再继续调整 bias。它提供均衡方向，并不要求一次就变成 5∶5。

>**Important**
> **在 DeepSeek 的这个机制中，bias 用于决定“选谁”；expert 输出的 gate weights 仍由原始 scores 得到。** Bias 不直接加进输出权重。[DeepSeek-V3 §2.1.2](https://arxiv.org/html/2412.19437v2#S2.SS1.SSS2)
