#AI #GPU

Data parallelism 是最直接的 [Parallelism](<Parallelism.md>)：不拆 model，只拆 batch。

>**Important** — Data parallelism 不拆 model，而是把一个 batch 拆给多个 ranks 并行计算。

```
global batch
→ split into local batches
→ 每个 rank 用完整 model 计算 local gradient
→ [All-Reduce](<All-Reduce.md>) 平均 gradients
→ 每个 rank 做相同的 optimizer update
```

## SGD example

Split the elements of B sized batch across M machines. Exchange gradients to synchronize
```math
\theta_{t+1} = \theta_t-\eta g
```
```math
g=\frac{1}{B}\sum_{i=1}^B\nabla_\theta \ell(\theta;x_i)
```
假设 batch 被均匀切成 ($M$) 份，rank ($m$) 的 local gradient 是：
```math
g_m=\frac{M}{B}\sum_{i\in\mathcal{B_m}}\nabla_\theta\ell(\theta;x_i)
```
于是
```math
g=\frac{1}{M}\sum_{m=1}^Mg_m
```

## How does it works
>**Note**
> Data parallelism 利用 batch gradient 可以分解求和这一点，用更多 GPU 并行处理数据；[All-Reduce](<All-Reduce.md>) 再把局部 gradients 还原成全局 gradient。

```
Compute：每个 rank 只处理 B/M 个 samples
Communication：每一步都要同步 gradients
Memory：每个 rank 仍保存完整 model states
```


Naive data parallelism 可以提高 training throughput，但**不能解决完整 model 放不进单张 GPU 的问题。**
## Memory scaling is absolutely none

Naïve data parallelism 只切了 batch，没有切 model states。所以每张 GPU 仍然保存完整的：
- parameters
- gradients
- **[Optimizer](<../../Transformer/Optimizer.md>) states** --- **真正占显存的**
- 当前 rank 所需的 activations

例如，一个模型训练状态需要 112 GB：
```
1 张 GPU：需要 112 GB
8 张 GPU naïve DP：每张仍需要 112 GB
```
集群虽然总共有很多显存，但 naïve DP 只是保存了 8 份相同副本，并没有把这些显存合起来使用。

关于 data parallelism 里的 Memory overhead 问题，可以看看 ZeRO
