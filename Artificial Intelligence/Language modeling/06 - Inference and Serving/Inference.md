#AI #LanguageModeling #Inference #Systems

Inference 是使用已经训练好的固定参数，对输入执行模型并得到预测或生成结果的阶段。

这份笔记主要关注 **LLM inference systems**：如何让 decoder-only language model 的生成过程高效运行并服务一个或多个 requests。

| [Autoregressive Decoding](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>) | 模型如何根据当前 context 选择 next token，并继续生成？           |
| --------------------------- | ----------------------------------------------- |
| Inference systems           | 整个 forward / generation workload 如何执行、优化、调度和服务？ |
# Landscape
#### Shows Up in many places
- Actual use (chatbots, code completion, agents, batch data processing)
- Model evaluation (e.g., on instruction following)
- Reinforcement learning (sample many generations, then apply score)

> **Important**
> Training is a one-time cost
> **Inference is repeated many times**

---
# Basic LLM Inference Flow

```text
request / prompt
→ prefill
→ first generated token
→ repeated decode steps (generation)
→ completed response
```

- **[Prefill](<Prefill.md>)**：处理 prompt 中已经给定的 tokens。
- **[Generation](<Generation.md>)**：每一步基于已有 context 生成一个新 token。
- **[KV Cache](<KV%20Cache.md>)**：复用 previous tokens 的 keys 和 values，避免每个 decode step 重算完整 prefix。

所以对于 inference 来说，输入并不是完整的 sequence 长度，一般会取：
```math
X\in\mathbb{R}^{B\times T\times D}
```
$T$ 是这一次 model execution 中，正在计算输出的 token positions 数量，不一定是当前完整 sequence/context 的长度。
- 在 [Prefill](<Prefill.md>) 阶段 $T=S$
- 在 [Generation](<Generation.md>) 阶段 $T=1$

---
# Make inference "**Fast**"

## What does "fast" mean (metrics)?

- [Time-to-first-token](<Time-to-first-token.md>) (TTFT): How long user waits before any generation happens (for interactive applications)
- [Latency](<Latency.md>) (seconds/token): how fast tokens appear for _one_ query (for interactive applications)
- [Throughput](<Throughput.md>) (tokens/second): how fast tokens appear for _many_ queries (for batch processing)

#### Sequential vs. Parallel

Training 可以用 [Parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Parallelism.md>) 来充分利用 [GPU](<../../GPU%20and%20NPU/GPU.md>) 的 compute，但是 inference 是 sequentially 的，所以没有并行的天然优势，[Why Training Parallelizes Across Positions but Generation Does Not](<../03%20-%20Training%20and%20Scaling/Training%20vs%20Inference.md#why-training-parallelizes-across-positions-but-generation-does-not>) 解释了这种现象

#### [Arithmetic Intensity in Inference](<Arithmetic%20Intensity%20in%20Inference.md>)

Inference 中的 [Arithmetic Intensity](<../../Fundamentals/Arithmetic%20Intensity.md>) 主要出问题在 [Generation](<Generation.md>) 的阶段，由 Attention layers 引入的

> **Summary** — Prefill is compute-bound, generation is memory-bound

#### 重复计算

假设 prompt 已有 100 个 tokens:
```
生成 token 101：重新计算 tokens 1...100
生成 token 102：重新计算 tokens 1...101
生成 token 103：重新计算 tokens 1...102
```

历史 token 已经固定，为什么还要重复计算它们的 K/V？

#### [KV Cache](<KV%20Cache.md>)

KV cache 解决 autoregressive inference 中的重复计算；它不能解决 token 间的顺序依赖，也没有消除低 arithmetic intensity，反而使 decode 的 memory-bound 特征更加明显。

> **Important** — **Inference is memory-bound**
#### [Latency](<Latency.md>) & [Throughput](<Throughput.md>) tradeoff

从直接的公式可以看出，这两个指标都和 [Batch Size](<../02%20-%20Language%20Modeling%20Basics/Batch%20Size.md>) 有关系

batch size ⬆️ $\longrightarrow$
- 一次并行生成更多 tokens $\longrightarrow$ parameter weights 可以被 batch 共享 $\longrightarrow$ throughput ⬆️
- 每条 request 有自己的 KV cache $\longrightarrow$ 总 KV-cache memory ⬆️ $\longrightarrow$ 每一步需要读取更多数据 $\longrightarrow$ 单步 latency ⬆️

```
小 batch → latency 较好 → throughput 较差

大 batch → latency 较差 → throughput 较好
```

但是这俩不是一直都是此消彼长的，如果降低 memory 两个都会变好

---
## How to make it Fast?

很重要的核心就是：**Inference is memory-bound**

所以怎么降低它的 memory 是 🚩
#### Shortcuts （有损）

一个很自然的想法💡：[Reduce KV cache size](<Reduce%20KV%20cache%20size.md>)
当然还可以做 [Quantization](<../../Quantization/Quantization.md>), [Model Pruning](<Model%20Pruning.md>)

#### Shortcuts + double checkout （无损）

先用一个比较小的 Draft model 低成本地把“未知未来”变成“候选已知序列”；然后再用 target model 利用并行 forward 验证候选，再通过 rejection-sampling 修正确保输出仍精确服从 target distribution：[Speculative Sampling](<Speculative%20Sampling.md>)

#### Serving

在线 serving 时，用户请求会随机到达，inference workload 不是规则矩形，而是参差不齐的：
```
A: ███████████████
B: ███
C: █████████████████████████
```

可以用 [Continuous Batching](<Continuous%20Batching.md>) 在每个 decode iteration 后移除已完成的 requests、补入新 requests，解决 static batch 的空槽、等待和 GPU utilization 下降。

vLLM 提出了一个改变 [KV Cache](<KV%20Cache.md>) 的 memory allocation/layout 的方法：[PagedAttention](<PagedAttention.md>)
