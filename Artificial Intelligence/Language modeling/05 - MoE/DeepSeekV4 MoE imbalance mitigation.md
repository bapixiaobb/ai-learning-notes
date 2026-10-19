#AI #LanguageModeling

DeepSeekV4 处理 [MoE](<Mixture%20of%20Experts.md>) 的 imbalance 方法是：

**主要用 [Per-expert bias](<Per-expert%20bias.md>) 调节整体负载，再用一个很小的 [sequence-wise balance loss](<Auxiliary%20load-balancing%20loss.md>)，避免单条 sequence 内出现极端不均衡。** [V4 §2.1](https://arxiv.org/html/2606.19348#S2.SS1)

dsv4 的处理把 batch 维度的和 sequence 维度的负载均衡分开来看的。这里统计整个 batch，和单独统计每条 sequence，看到的问题可能不同
## Example

>**Example** — 假设两个 experts，top-1
>|统计范围|A 收到的 tokens|B 收到的 tokens|
>|---|---|---|
>|Sequence 1：10 个 tokens|9|1|
>|Sequence 2：10 个 tokens|1|9|
>|**整个 batch：20 个 tokens**|**10**|**10**|

#### Per-expert bias 处理

它根据**整个 batch 的负载调整** bias。这里 A、B 各收到 10 个 tokens，整体已经均衡，因此这次没有必要因为负载差异而升降它们的 bias。

但这并不代表每条 sequence 都均衡：Sequence 1 几乎只用 A，Sequence 2 几乎只用 B。

#### Sequence-wise balance loss

用[这里面的例子](<Auxiliary%20load-balancing%20loss.md#example>)：

假设 Sequence 1 内： $$f_A=9/10=0.9,\qquad f_B=1/10=0.1,\qquad\qquad P_A=0.58,\qquad P_B=0.42$$按前面学过的简化形式：
```math
L_1=1.8P_A^{(1)}+0.2P_B^{(1)}
```
这里 $P_A^{(1)}$ 表示 **只对 Sequence 1 的 tokens 取平均**的 A 概率；上标 `(1)` 表示第一条 sequence。

把一点概率从 A 移给 B，会让 \(L_1\) 降低。因此，这个 loss 对 **Sequence 1 的 tokens** 提供的推动方向是：**少偏向 A，多给 B 一些概率。**

对 Sequence 2，情况反过来：B 热门。

它自己的统计结果是 $f_A=0.1,\ f_B=0.9$，因此：
```math
L_2=0.2P_A^{(2)}+1.8P_B^{(2)}
```
这个 loss 对 **Sequence 2 的 tokens** 提供的推动方向是：**少偏向 B，多给 A 一些概率。**

最后，在这个例子里，把两个 loss 取平均，再加入训练目标：

```math
\text{total loss} = \text{LM loss} + \lambda\frac{L_1+L_2}{2}
```

然后正常 backward。虽然 router 参数是共享的，但两条 sequence 的输入不同，各自产生的 gradient 会一起用于更新它。
