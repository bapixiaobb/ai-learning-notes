#LanguageModeling #Systems #Optimization

> **Note** — Scaling Law 是研究某个 scale variable 增大时，模型的 loss / performance 如何按照可预测的数量关系变化
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

通常近似固定 model class 和 training setup，只改变 dataset size / training tokens $D$，然后观察 held-out loss 或 error 如何随 $D$ 变化。

因此，data scaling 研究的是 $L(D)$，而改变参数量 $N$ 的 model scaling 研究的是 $L(N)$。如果同时研究 $N$ 和 $D$，则需要 joint model-data scaling law。

---
## Scaling laws for model engineering

对于 model scaling 来说，我们有问题要问
> **Question** — How can we efficiently design huge [LMs](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20%28LLM%29.md>) ?
> [Hyperparameter Choice](<Hyperparameter%20Choice.md>)

> **Question** — How should we allocate our limited resouces?
>> Train models longer vs train bigger models?
>>Collect more data vs get more GPUs?

---
## Dense Transformer Training Compute

在 dense Transformer training 中，training compute 常用粗略近似：
```math
C \approx 6ND
```
其中 $N$ 是参数量，$D$ 是训练 token 数。这里的常数 $6$ 及其适用范围见 [Training Compute - 6ND](<Training%20Compute%20-%206ND.md>)。


> **Note** — **与 Resource Accounting 的关系**
> [Resource Accounting](<Resource%20Accounting.md>) 关注训练需要多少 compute、memory、bandwidth 和时间；Scaling law 关注这些资源投入后能换来多少 loss improvement。
>
> 前者回答“训练要花多少资源”，后者预测“增加资源可能带来多少边际改善”。是否值得投入，还取决于训练和部署目标。


> **Note** — **与 [Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 的关系**
> Language model 的 loss 通常是平均到每个 token 的 [Cross Entropy Loss](<../01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>) 或 negative log-likelihood。
>
>在 autoregressive language modeling 中，可以写成：
>```math
>L(\theta)
>=
>-\frac{1}{T}\sum_{t=1}^{T}
>\log p_\theta(x_t\mid x_{<t})
>```
>Scaling law 所研究的通常是 held-out data 上的这种 per-token loss 如何随实验规模变化。
