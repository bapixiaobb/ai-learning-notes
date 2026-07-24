#AI #LanguageModeling

Batch size 是一个 [training hyperparameter](<Model%20Hyperparameters.md>)，指一次参数更新使用多少 training examples / tokens。

在 language model training 中，更常见的是按 tokens 理解 batch size：

```math
\text{batch tokens}
=
\text{batch size}
\times
\text{sequence length}
```

如果使用 [gradient accumulation](<../02%20-%20Training%20and%20Scaling/Gradient%20Accumulation.md>)，则 effective batch size 可以写成：

```math
\text{effective batch size}
=
\text{microbatch size}
\times
\text{gradient accumulation steps}
\times
\text{number of devices}
```

>**Note**
>Batch size 既是 [training recipe](<../02%20-%20Training%20and%20Scaling/Training%20Recipe.md>) 的一部分，也和 [Systems for Language Models](<../03%20-%20GPU%20and%20Systems/Systems%20for%20Language%20Models.md>) 强相关。
>
>较大的 batch size 通常能提高硬件利用率，但也会增加 memory pressure，并可能影响 optimization dynamics。

---
## Critical batch size

在 [Scaling Law](<../02%20-%20Training%20and%20Scaling/Scaling%20Law.md>) 中会从 [Hyperparameter Choice](<../02%20-%20Training%20and%20Scaling/Hyperparameter%20Choice.md>) 了解到怎么选择 batch size，batch size 的选择有一个最优
![optimal batch size.png](<../../attachments/optimal%20batch%20size.png>)
⬆️ known to have strong diminishing returns past a certain point

Critical batch = min number of examples before diminishing returns

这里有一个公式

先固定一个 target loss，例如：
```
目标：把 train loss 降到 4.0
```
然后用不同 batch size 训练，并记录：

- $S$：达到 target loss 需要多少 [optimizer steps](<../../Transformer/Optimizer.md>)；
- $E$：达到 target loss 总共处理多少 examples/tokens。

两者满足：
```math
E=B\times S
```
小 batch：
```
每一步样本少
→ 需要很多 steps
→ 但总共浪费的 examples 较少
```
大 batch：
```
每一步样本多
→ 需要较少 steps
→ 但可能处理很多没有带来额外收益的 examples
```
**critical batch size** 就是中间的转折点：****再继续增大 batch，收益开始明显递减的 batch size。****

因此存在两个极端：
- $S_{\min}$：理论上最少的 steps；
- $E_{\min}$：理论上最少的 examples。

在“少走 steps”和“少浪费 examples”之间取一个平衡：
```math
B_{\mathrm{crit}}=\frac{E_{\min}}{S_{\min}}
```

```
step efficiency ↔ sample efficiency
```
