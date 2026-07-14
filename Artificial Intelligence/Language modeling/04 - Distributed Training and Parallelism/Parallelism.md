#AI #LanguageModeling #GPU

从数学上看，我们仍然在训练同一个 model、计算同一个 gradient update。
但当一次 training step 所需的 compute 或 memory 超过单张 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 能提供的资源时，就需要把这个计算分配到多个 devices。

所以 Parallelism 的核心问题很简单：**多张 GPU 一起训练时，到底切什么。**

它是 [Systems for Language Models](<../03%20-%20GPU%20and%20Systems/Systems%20for%20Language%20Models.md>) 中的一种 execution plan：[Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 先暴露资源瓶颈，Parallelism 再决定怎样拆分 computation 和 model states。

一旦训练被切开，不同 ranks 手里就只有局部信息，所以需要 communication 把缺失的信息补回来。

>**Note**
> Parallelism = 怎么切。
> Collective communication = 切完之后怎么交换 / 聚合信息。

# [Data parallelism](<Data%20parallelism.md>)

切 batch / data。

每个 rank 有完整 model，只看不同 batch shard。
backward 后每个 rank 的 gradients 不同，所以用 [All-Reduce](<All-Reduce.md>) 平均 gradients。

# Model Parallelism
## Tensor parallelism

切 layer 里的矩阵 / hidden dimension。

每个 rank 只持有某一层参数或 activation 的一部分。
forward 常需要 [All-Gather](<All-Gather.md>) activations；
backward 常对应 [Reduce-Scatter](<Reduce-Scatter.md>) gradients。
## Pipeline parallelism

切 layers / depth。

每个 rank 持有连续几层。
rank 之间主要 send / recv activations；
micro-batches 用来减少 pipeline bubbles。

---
# 🔗
[GPU Communication Topology](<GPU%20Communication%20Topology.md>)
[NVIDIA Collective Communication Library](<NVIDIA%20Collective%20Communication%20Library.md>)
