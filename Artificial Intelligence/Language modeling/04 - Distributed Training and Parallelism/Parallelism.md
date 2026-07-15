#AI #LanguageModeling #GPU

从数学上看，我们仍然在训练同一个 model、计算同一个 gradient update。
但当一次 training step 所需的 compute 或 memory 超过单张 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 能提供的资源时，就需要把这个计算分配到多个 devices。这里涉及到 [GPU Communication Topology](<./GPU%20Communication%20Topology.md>)

所以 Parallelism 的核心问题很简单：**多张 GPU 一起训练时，到底切什么。**

它是 [Systems for Language Models](<../03%20-%20GPU%20and%20Systems/Systems%20for%20Language%20Models.md>) 中的一种 execution plan：[Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 先暴露资源瓶颈，Parallelism 再决定怎样拆分 computation 和 model states。

一旦训练被切开，不同 ranks 手里就只有局部信息，所以需要 communication 把缺失的信息补回来。这个 communication 借助底层指令来完成 [NVIDIA Collective Communication Library](<./NVIDIA%20Collective%20Communication%20Library.md>)

>**Note**
> Parallelism = 怎么切。
> Collective communication = 切完之后怎么交换 / 聚合信息。

---
# [Data parallelism](<./Data%20parallelism.md>)：切 batch / data。

每个 rank 有完整 model，只看不同 batch shard。
backward 后每个 rank 的 gradients 不同，所以用 [All-Reduce](<./All-Reduce.md>) 平均 gradients。

>**Question** — 为什么只靠 Data Parallelism 不够？
>
> **Compute scaling**
> Data parallelism 只能沿 batch dimension 使用更多 GPUs。固定 global batch 时，GPU 数受到 batch size 限制；继续扩大 global batch 又会在 critical batch size 后产生 diminishing returns。
>
> **Memory scaling**
> [ZeRO](<./ZeRO.md>) 可以 shard parameters、gradients 和 optimizer states，但每个 data-parallel rank 仍要为自己的 local batch 执行普通 [computation graph](<../../Neural%20Networks/Computational%20Graph.md>)，并保存相应 [activations](<../../Neural%20Networks/Activations.md>)。ZeRO 本身不会切分 activation 的 layer、hidden 或 sequence dimensions。

Data parallelism 切 batch，能够使用的 GPUs 受 batch size 限制；model parallelism 改为切 layers、matrices 或 sequence，从其他维度继续分配 compute 和 memory，并开始在 ranks 之间传输 activations。

---
# [Model Parallelism](<./Model%20Parallelism.md>)
## [Tensor Parallelism](<./Tensor%20Parallelism.md>)：切 layer 里的矩阵 / hidden dimension。

每个 rank 只持有同一层 parameters / activations 的一个 shard。
根据矩阵的具体切法，[forward](<../../Neural%20Networks/Forward%20Propagation.md>) 和 [backward](<../../Neural%20Networks/Backpropagation.md>) 可能使用 [All-Gather](<./All-Gather.md>)、[Reduce-Scatter](<./Reduce-Scatter.md>) 或 [All-Reduce](<./All-Reduce.md>) 来组合 activations 及其 gradients。
## [Pipeline Parallelism](<./Pipeline%20Parallelism.md>)：切 layers / depth。

每个 rank 持有连续几层。  rank 之间主要 send / recv activations；  micro-batches 用来减少 pipeline bubbles。
## [Expert parallelism](<./Expert%20parallelism.md>)

---
# How to decide: [Parallel Strategies](<./Parallel%20Strategies.md>)

这一页回答“可以切什么”；[Parallel Strategies](<./Parallel%20Strategies.md>) 进一步回答“面对具体 model 和 hardware，应该怎样组合这些切法”。

它本质上是一个 resource-allocation problem：先根据 [Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 找到 bottleneck，用最少的 model parallelism 让 model / activations 装得下，再根据 [GPU Communication Topology](<./GPU%20Communication%20Topology.md>) 安排 communication groups，最后把剩余 GPUs 尽量用于 [Data parallelism](<./Data%20parallelism.md>)。

![LLM parallelism table.png](<../attachments/LLM%20parallelism%20table.png>)
