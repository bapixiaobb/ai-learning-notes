#AI #LanguageModeling #GPU

Parallelism 的核心问题很简单：**多张 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 一起训练时，到底切什么。**

一旦训练被切开，不同 ranks 手里就只有局部信息，所以需要 communication 把缺失的信息补回来。

> **Note**
> Parallelism = 怎么切。  
> Collective communication = 切完之后怎么交换 / 聚合信息。

# Data parallelism

切 batch / data。

每个 rank 有完整 model，只看不同 batch shard。  
backward 后每个 rank 的 gradients 不同，所以用 all-reduce 平均 gradients。

# Tensor parallelism

切 layer 里的矩阵 / hidden dimension。

每个 rank 只持有某一层参数或 activation 的一部分。  
forward 常需要 all-gather activations；  
backward 常对应 reduce-scatter gradients。

# Pipeline parallelism

切 layers / depth。

每个 rank 持有连续几层。  
rank 之间主要 send / recv activations；  
micro-batches 用来减少 pipeline bubbles。

---
# 🔗
[GPU Communication Topology](<./GPU%20Communication%20Topology.md>)
[NVIDIA Collective Communication Library](<./NVIDIA%20Collective%20Communication%20Library.md>)
