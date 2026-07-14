#LanguageModeling #Systems

Resource Accounting 可以理解成：**在真正运行之前，先给一次 training step 记账。**

数学上我们只写：

```math
\theta_{t+1}
=
\theta_t-\eta\,g_t
```

但在机器上完成这个 update，需要执行 forward / backward，保存中间 tensor，并在 memory 和 compute units 之间搬运数据。

因此真正需要数清三笔账：

    做多少计算（compute）
    存多少数据（memory）
    搬多少数据（data movement / communication）

# 1. Compute：要算多少

Transformer 的主要计算来自 matrix multiplication，例如 attention projections 和 MLP。

对于 dense Transformer，完整训练的计算量常粗略写成：

```math
\text{Training FLOPs} \approx 6ND
```

其中 $N$ 是 parameters 数量，$D$ 是训练过的 tokens 数量。更详细的来源见 [Training Compute - 6ND](<Training%20Compute%20-%206ND.md>)。

这个数字告诉我们总工作量，但不能直接告诉我们训练会有多快。真实速度还取决于 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 的 throughput，以及 compute units 是否在等待数据。

# 2. Memory：要存什么

训练时显存不只存 parameters：

| 内容 | 为什么需要 |
|---|---|
| parameters | forward / backward 使用的模型权重 |
| gradients | optimizer 更新 parameters 所需 |
| optimizer states | Adam 保存的一阶、二阶统计量等 |
| activations | backward 计算梯度时需要的中间结果 |

所以“模型参数能放进显存”不等于“模型能够训练”。

如果放不下，可以改变执行方式：

- low precision：让每个 tensor 占更少 bytes；
- [Recomputation](<Recomputation.md>)：少存 activations，backward 时重新计算；
- ZeRO / FSDP：把 model states 分到多个 devices；
- tensor / pipeline parallelism：把模型本身拆开。

# 3. Data Movement：数据要经过哪里

即使 compute 和 memory 容量都够，数据也必须及时送到 compute units。

单张 GPU 内部：

    HBM ↔ shared memory / registers ↔ compute

如果 bytes 搬得很多、计算却很少，operator 就可能是 [GPU Memory Bound](<../03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>)。[Tiling](<../03%20-%20GPU%20and%20Systems/Tiling.md>)、[Operator fusion](<../03%20-%20GPU%20and%20Systems/Operator%20fusion.md>) 和 [Memory Coalescing](<../03%20-%20GPU%20and%20Systems/Memory%20Coalescing.md>) 都是在改善这笔账。

多张 GPU 之间：

    GPU ↔ GPU
    node ↔ node

这时还要计算 communication volume，并结合 [GPU Communication Topology](<../04%20-%20Distributed%20Training%20and%20Parallelism/GPU%20Communication%20Topology.md>) 判断传输速度。

# Resource Accounting 如何导向 Parallelism

>**Question**
> 给定 model、batch 和 hardware，单张 GPU 是否放得下？如果放得下，是否能在合理时间内完成训练？

如果答案是否定的，就需要 [Parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Parallelism.md>)。

Parallelism 决定把什么拆开：

| 被拆分的数学对象 | Parallelism |
|---|---|
| batch / gradient sum | data parallelism |
| matrix / hidden dimension | tensor parallelism |
| layers / function composition | pipeline parallelism |
| parameters、gradients、optimizer states | ZeRO / FSDP |

但拆分会产生新的 communication。因此不能只问“能不能切”，还要比较：

```math
\text{节省的 memory / 增加的 compute}
\quad\text{vs.}\quad
\text{新增的 communication}
```

>**Summary**
> Resource Accounting 不是一堆 performance 指标，而是沿着一次 training step，数清 **算什么、存什么、搬什么**。
> 它负责暴露资源瓶颈；[Parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Parallelism.md>)、kernel optimization、low precision 和 recomputation 则是不同的解决办法。

---
# 🔗

[Systems for Language Models](<../03%20-%20GPU%20and%20Systems/Systems%20for%20Language%20Models.md>)
[Transformer](<../../Transformer/Transformer.md>)
[Training Recipe](<Training%20Recipe.md>)
[GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>)
[Parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Parallelism.md>)
[FLOPs](<../03%20-%20GPU%20and%20Systems/FLOPs.md>)
[Model FLOPs Utilization](<../03%20-%20GPU%20and%20Systems/Model%20FLOPs%20Utilization.md>)
[Scaling Law](<Scaling%20Law.md>)
