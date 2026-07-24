#AI #LanguageModeling #GPU

[Parallelism](<Parallelism.md>) 解释多张 GPU 训练时“可以切什么”；Parallel Strategies 关心的是：**面对具体 model 和 hardware，应该怎样组合这些切法。**

[Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 先告诉我们瓶颈来自 parameters、[Activations](<../../Neural%20Networks/Activations.md>)、compute 还是 communication；parallel strategy 再决定把 GPU 分配到 model width、depth、sequence、experts 或 data 中。

>**Important**
>这些策略通常不是互斥选项，而是共同组成一个 execution plan。
>
>核心原则是：**先用最少的 model parallelism 让训练装得下，再把剩余 GPU 尽量用于 data parallelism。**

## What each strategy buys

| 策略                           | 切什么                                  | 解决的问题                                           | 主要代价                                                   |
| ---------------------------- | ------------------------------------ | ----------------------------------------------- | ------------------------------------------------------ |
| [DP](<Data%20parallelism.md>)     | batch / data                         | 增加训练吞吐量                                         | gradient synchronization；受 useful global batch size 限制 |
| [ZeRO](<ZeRO.md>) / FSDP              | model states                         | parameters、gradients、optimizer states 占内存       | 需要按阶段 gather / communicate model states                |
| [TP](<Tensor%20Parallelism.md>)   | layer 内的 matrices / hidden dimension | 一层太宽，一张卡算不下                                     | 高频 activation communication；local matmul 会变小           |
| [PP](<Pipeline%20Parallelism.md>) | Transformer layers / depth           | 整个 model 装不下                                    | pipeline bubbles；需要足够的 microbatches                    |
| CP                           | sequence dimension                   | long context 的 activation / attention memory 太大 | sequence shards 之间需要通信                                 |
| [EP](<Expert%20parallelism.md>)   | MoE experts                          | 把多个 expert MLP 分到不同 GPUs                        | latency-sensitive All-to-All 和 load balance            |

## How to choose

### 1. Identify what does not fit

- 主要是 model states：先考虑 [ZeRO](<ZeRO.md>) / FSDP。
- 单层 matrix 太大：使用 [Tensor Parallelism](<Tensor%20Parallelism.md>)。
- 整个 model 太深或参数太多：增加 [Pipeline Parallelism](<Pipeline%20Parallelism.md>)。
- sequence 太长：增加 CP。
- model 是 [MoE](<../05%20-%20Architectures%20and%20MoE/Mixture%20of%20Experts%20(MoE).md>)：expert MLP 优先使用 [Expert parallelism](<Expert%20parallelism.md>)；attention 仍可以使用 TP。

如果主要问题是 saved [Activations](<../../Neural%20Networks/Activations.md>)，[Recomputation](<../02%20-%20Training%20and%20Scaling/Recomputation.md>) 可以用额外 compute 换 memory。省下的 memory 可以转化为更大的 local batch，有时反而会提高 [hardware utilization](<../03%20-%20GPU%20and%20Systems/Model%20FLOPs%20Utilization.md>)。

### 2. Match communication to [GPU Communication Topology](<GPU%20Communication%20Topology.md>)

- TP 和 EP communication 很频繁，优先放在 NVLink / NVSwitch 这样的 fast domain；GPU 上的 TP 通常控制在 8 以内。
- PP 只在 stage boundaries 传 boundary activations / gradients，通信相对少，更适合跨 node 的较慢 links。
- DP 使用剩余 GPUs 扩大吞吐，但不能无限扩大 global batch；超过 [critical batch size](<../01%20-%20Language%20Modeling%20Basics/Batch%20Size.md>) 后会有 diminishing returns。

### 3. Use the remaining GPU budget for DP

对于普通 dense model，可以先粗略看成：

```
number of GPUs ≈ DP × TP × PP × CP
```

这些数字不是相加，而是不同 parallel dimensions 组成的 device mesh。

例如 Llama 3 405B 主训练阶段：

```
TP = 8, PP = 16, DP = 128, CP = 1
8 × 16 × 128 × 1 = 16384 GPUs
```

TP 和 PP 是为了让 model 能够执行；model 装下之后，剩余 GPU 用 DP 提高 throughput。

## MoE needs two parallel views

MoE 只替换 Transformer block 中的 MLP / FFN，不替换 attention。因此两部分想要的 parallelism 不一样：

```
Attention: high TP / CP can be useful
MoE MLP:  high EP, low ETP is usually preferable
```

Megatron 的 MoE Parallel Folding 允许同一批 GPUs 使用两套 logical groups：

```
Attention: TP × CP × DP × PP
MoE MLP:   ETP × EP × EDP × PP
```

Attention 执行时进入 attention 的 TP / CP groups；进入 expert MLP 时，同一批 ranks 重新按照 ETP / EP / EDP 分组。这样不会因为 attention 需要高 TP，就把每个 expert 的 SwiGLU matrices 也切得很小。

## What real training runs show

| Model / experiment             | Parallel strategy                                                        | What it shows                                                                         |
| ------------------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- |
| Narayanan et al. scaling sweep | 先最大化 DP；TP 增长到约 8 后停止；更大规模继续增加 PP                                        | TP 受 fast-link domain 和 small-matmul efficiency 限制；PP 需要足够大的 batch 填满 pipeline        |
| OLMo 7B                        | FSDP                                                                     | 对相对较小的 dense model，FSDP 本身可以扩展到很多 GPUs，不一定需要复杂 TP / PP                                |
| DeepSeek V1 / Yi               | ZeRO-1 + TP（/ SP）+ PP                                                    | dense large model 的经典组合                                                               |
| DeepSeek V3                    | PP + EP64                                                                | MoE 用 EP 代替 expert MLP 中的大量 TP；通过 specialized communication 和 scheduling 扩大 EP domain |
| Llama 3 405B                   | main pretraining: TP8 + PP16 + DP128；long-context extension 时提高 CP、降低 DP | dense model 使用 TP 而不是 EP；训练阶段的 bottleneck 变化时，parallel configuration 也会变化             |
| Gemma 2                        | FSDP + TP / SP，没有重点使用 PP                                                 | TPU toroidal mesh 使更大范围的 TP 可行；strategy 受 hardware topology 影响                        |
| Mixtral 8×22B                  | EP8 + PP4；attention 使用 TP4                                               | EP 和 TP 可以在同一个 block 中分别服务 expert MLP 与 attention                                     |
| Nemotron 3 Super               | large EP + CP                                                            | MoE 与 long-context 同时存在时，要分别为 experts 和 sequence 分配 GPUs                              |
| Qwen 3                         | EP32 + PP8；attention 使用 TP2                                              | 现代 MoE 可以使用较大 EP，同时保留较小 TP 处理 attention                                               |

Llama 3 训练中频繁发生 GPU failures 还说明：规模变大后，系统不仅需要高 utilization，也需要 checkpoint、failure recovery 和 redundancy。

>**Summary** — My Understanding
>Parallel strategy 不是背一套固定配置，而是一个 resource-allocation problem：
>
>1. 根据 model architecture 和 [Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 找到当前 bottleneck；
>2. 使用最少的 TP / PP / CP / EP 让 model 和 activations 装得下；
>3. 把 communication-heavy groups 放在更快的 [links](<GPU%20Communication%20Topology.md>) 上；
>4. 将剩余 GPU 尽量用于 [Data parallelism](<Data%20parallelism.md>)；
>5. 通过 batch size、microbatches、[Recomputation](<../02%20-%20Training%20and%20Scaling/Recomputation.md>) 和 communication overlap 提高 utilization。
