#LanguageModeling #Systems #Optimization

>**Note** — Scaling Law 是研究某个 scale variable 增大时，模型的 loss / performance 如何按照可预测的数量关系变化
>核心是，训练多个规模的模型
>
>|参数量 \(N\)|Validation loss|
>|---|---|
>|100M|3.2|
>|300M|2.9|
>|1B|2.6|
>如果它们近似满足 $L(N)=L_\infty +AN^{-\alpha}$ 我们就可以拟合 $A$, $\alpha$, $L_\infty$，进而预测 10B 模型的 loss

⚠️ Scaling law 让我们利用一系列低成本的小规模实验，拟合 performance 与 scale 之间的数量规律，从而预测大规模训练的表现并指导工程决策；它预测的是可外推的规律，不是简单把小模型结果原样复刻到大模型。

---
## [Data Scaling Laws](<Data%20Scaling%20Laws.md>)

通常近似固定 model class 和 training setup，只改变 dataset size / training tokens $D$，然后观察 held-out loss 或 error 如何随 $D$ 变化。因此，data scaling 研究的是 $L(D)$

---
## Scaling laws for model engineering

改变参数量 $N$ 的 model scaling 研究的是 $L(N)$
#### Hyperparameters Choice
对于 model scaling 来说，我们有问题要问
>**Question** — How can we efficiently design huge [LMs](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>) ?
> The effect of hyperparameters on big LMs can be predicted before training!
> 所以 [Hyperparameter Choice](<Hyperparameter%20Choice.md>) 很重要

The scaling law based design procedure.

1. Train a few smaller models
2. Establish a scaling law (e.g. ADAM vs SGD scaling law)
3. Select optimal hyperparameters based on the scaling law prediction.

#### 但是 scaling law 能预测的东西是有边界的

[Scaling Law Regime](<Scaling%20Law%20Regime.md>): Scaling law 只能直接预测它所拟合的指标。

---
## How should we allocate our limited [resources](<Resource%20Accounting.md>)?

同时考虑 $N$ 和 $D$

>**Question**
>Train models longer vs train bigger models?
>Collect more data vs get more GPUs?

现实中，loss 同时由 dataset size 和 model size 两者决定：
```math
L(N,D) \approx A D^{-\alpha} + B N^{-\beta} + L_\infty
```
- $D$：data/token 数量
- $N$：model parameters
- $AD^{-\alpha}$：因为 data 不够产生的 loss
- $BN^{-\beta}$：因为 model 不够大产生的 loss
- $L_\infty$：两者都很大后仍然存在的最低 loss

>**Question** — Do we need more data or bigger models?
>![model size vs dataset size.png](<../../attachments/model%20size%20vs%20dataset%20size.png>)
>Clearly, lots of data is wasted on small models

#### Train models longer vs train bigger models?

在 dense Transformer training 中，training compute 常用粗略近似：
```math
C \approx 6ND
```
其中 $N$ 是参数量，$D$ 是训练 token 数。这里的常数 $6$ 及其适用范围见 [Training Compute - 6ND](<Training%20Compute%20-%206ND.md>)。

固定 compute 时，$N$ 和 $D$ 不能一起随便增加。于是问题变成：

>**Question** — 我是应该训练“大模型 + 少量 tokens”，还是“小模型 + 更多 tokens”？

Scaling law 领域有一个非常有名的发现，是 2022 年 DeepMind 的 **Training Compute-Optimal Large Language Models**。[Chinchilla](<Chinchilla%20Scaling%20Law.md>)  是论文里训练的模型名

>**Note** — Chinchilla：模型和 data 必须平衡，很多大模型其实没有被充分训练。

Chinchilla 用三条不同路线，Compute-Optimal Model or Data Allocation 回答同一个问题：给定 Compute $C$，$N$ 取多大，$D$ 取多少，loss 最低

---

## Difference?

>**Question** — 为什么不同的方法（Kaplan 和 Chinchilla）都是研究 $C=6ND$ 但是结论会差这么多？
#### Parameter counting 不同

Kaplan 没有把 embedding 和最后的 LM head 等 parameters 全部计入 $N$。

尤其小模型中，embedding/LM head 占比更大，所以影响不是简单地把所有点平移一下。
#### 小模型的 warmup 不合适

一些小模型的 training run 很短，但 warmup 却相对太长。结果是：小模型还没在合适的 step size 下充分训练，实验就结束了。

这些小模型因此显得比实际更差。Scaling law 又依靠这些小模型预测大模型，所以起点测歪了，外推也会歪。
#### Batch size 没有随规模调整

Kaplan 对不同大小的模型使用一个固定的大 batch size，但这个 batch size 对小模型不一定合适。

重新调整 parameter counting、warmup、optimizer 和 batch size 后，结果会逐渐从 Kaplan 靠近 Chinchilla。

>**Important** — 如果小规模实验没有正确调参，那么你只是在准确地 extrapolate 一个糟糕的 recipe。

---

## Chinchilla-optimal 不一定是我们真正想要的

真实产品关心的是：
```math
\text{total cost} = \text{one-time training cost} + \text{大量 inference cost}
```

一个大模型可能 training-compute optimal，但每次 inference 都更贵、更慢、占更多显存。

所以生产环境经常选择：
- 更小的模型；
- 用远多于 20 tokens/parameter 的 data 训练更久；
- 换取更便宜的长期 inference。

这称为 “overtrained”，但不是我们通常说的过拟合。它的意思是：

>**Note** — 相对于 Chinchilla 的 training-compute optimal point，模型训练了更多 tokens。

因为 training 只付一次钱，而 inference 可能执行几十亿次。预计使用次数越多，就越值得提前多花 training compute，把模型做小一点。
