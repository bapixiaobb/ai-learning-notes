#LanguageModeling #LLM #Transformer #StudyMap

这份笔记是当前 Language Modeling 笔记的总览地图。它把零散概念按学习主线整理成：

```text
objective
-> data / tokens
-> model architecture
-> forward pass / loss
-> training recipe
-> inference
-> scaling / resource accounting
-> GPU systems optimization
-> advanced architectures
```

## 1. One-Sentence Core

[[Language Modeling]] 的核心是：

> 给定前面的 tokens，学习一个概率分布来预测下一个 token。

数学上，autoregressive language model 建模：

```math
p_\theta(x_1,\dots,x_T)
=
\prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})
```

训练时通常最大化真实 next token 的 likelihood，等价于最小化 [[Cross Entropy Loss]]。

## 2. 最重要的区分

| Concept                         | 它回答的问题                          | 典型笔记                                                                                  |
| ------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------- |
| [[Language Modeling]]           | 模型要学什么目标？                       | next-token probability                                                                |
| [[Transformer]]                 | 用什么 neural architecture 实现这个目标？ | self-attention, block, residual stream                                                |
| [[Language Model Architecture]] | 模型内部有哪些 design choices？         | norm, position, [[MLP]], attention variant                                            |
| [[Training Recipe]]             | 参数怎么被训练出来？                      | [[Optimizer]], [[Learning Rate Schedule]], batch size                                 |
| [[Resource Accounting]]         | 训练和推理要花多少 compute / memory？     | [[FLOPs]], memory, activation, optimizer states                                       |
| [[GPU Memory Bound]]            | 怎么让 workload 在硬件上跑得快？           | [[Tiling]], [[Operator fusion]], [[Memory Coalescing]], [[Low precision computation]] |

一个常见混淆是：

> Language modeling 是 objective；Transformer 是 architecture；training recipe 是训练方法；systems 是高效执行的方法。

## 3. Pipeline

一个 decoder-only language model 的数据流可以写成：

```text
raw text
-> [[Tokenization]]
-> token IDs
-> [[Token Embedding]]
-> positional information, usually [[Rotary Position Embedding]]
-> stacked [[Transformer Block]]
-> final hidden states
-> language modeling head
-> logits
-> [[Softmax]]
-> next-token probability
-> [[Cross Entropy Loss]]
```

其中 [[Causal Mask]] 保证每个位置只能看到 prefix，不能偷看 future tokens。

## 4. Objective And Loss

这组笔记的核心链条是：

- [[Next-token prediction]]：理论目标是预测下一个 token。
- [[Language Modeling]]：把整个 token sequence 的概率分解成 conditional probabilities。
- [[Cross Entropy Loss]]：训练时惩罚模型没有把足够高的概率分给真实 next token。
- likelihood view：最小化 cross entropy 等价于最大化训练数据 likelihood。

可以把每个 token position 看成一个 vocabulary-size 的多分类问题：

```math
\mathcal{L}_t
=
-\log p_\theta(x_t \mid x_{<t})
```

整段序列通常对所有 token positions 求平均。

## 5. Architecture 主线

你的 architecture 笔记已经形成了很清楚的层次：

### 5.1 Transformer 是主体网络

[[Transformer]] 是 attention-based sequence model。它的关键不是 recurrence，而是让 token positions 通过 [[Self-Attention]] 交换信息。

Transformer 本身不等于 language modeling。GPT-style LM 更准确地说是：

```text
[[Decoder-Only Transformer]]
+
autoregressive next-token prediction objective
```

### 5.2 Modern LLM 通常是 decoder-only

[[Decoder-Only Transformer]] 的关键是：

- causal self-attention；
- 每个位置只能 attend to prefix；
- 训练时可以并行处理整段 sequence；
- 推理时 autoregressively 一个 token 一个 token 生成；
- 通常使用 KV cache 降低重复计算。

### 5.3 Transformer block 是重复单元

[[Transformer Block]] 可以理解为：

```text
residual stream
-> attention update
-> residual stream
-> MLP update
-> residual stream
```

其中：

- [[Self-Attention]] / [[Causal Attention]] 负责 token mixing；
- [[MLP]] 负责 per-token nonlinear processing；
- [[Residual Connection]] 保留主 hidden state path；
- [[Normalization]] / [[RMSNorm]] 稳定 hidden state scale；
- [[Pre-Norm Transformer]] 是现代 LLM 常见 block layout。

### 5.4 Llama-style 是 modern decoder-only 的代表范式

[[Llama-style Architecture]] 不是说所有模型都是 Llama，而是代表一组常见 choices：

- [[Decoder-Only Transformer]]
- [[Causal Attention]]
- [[Rotary Position Embedding]]
- [[RMSNorm]]
- [[SwiGLU]]
- [[Pre-Norm Transformer]]
- often [[Grouped Query Attention]]

## 6. Attention 主线

Attention 相关笔记可以按这个顺序复习：

1. [[Self-Attention]]：同一个 sequence 内部 token positions 交换信息。
2. [[Query Key Value]]：Q 表示当前 token 要找什么，K 表示其他 token 提供什么索引，V 表示真正被聚合的信息。
3. [[Artificial Intelligence/Transformer/Multi-Head Attention]]：多个 attention heads 学不同的关系子空间。
4. [[Causal Attention]]：加上 causal constraint 后只允许看 prefix。
5. [[Causal Mask]]：实现上把 future positions 的 attention score 变成 $-\infty$。
6. [[Flash Attention]] / [[Online Softmax]]：通过 tiling 和 online softmax 减少 HBM traffic。

核心直觉：

> Attention 不是直接输出最终答案，而是在 residual stream 中写入 context-dependent update。

## 7. Training 主线

[[Training Recipe]] 这组笔记回答的是：architecture 固定后，如何把参数训练出来。

关键组成：

- objective：next-token prediction / cross entropy；
- [[Optimizer]]：常见是 AdamW；
- [[Learning Rate Schedule]]：warmup + decay；
- batch size：通常按 batch tokens 理解；
- [[Gradient Accumulation]]：用多个 microbatches 模拟更大的 effective batch；
- weight decay：控制参数范数；
- dropout：训练时 regularization，inference 时关闭；
- gradient clipping：限制 update 幅度，提高稳定性；
- data mixture / token count：决定模型学到什么和学多充分。

这里最重要的区分：

> Training recipe 不改变 forward pass 的结构，但会极大影响最终模型性能和稳定性。

## 8. Training vs Inference

[[Training vs Inference]] 是你当前笔记里很重要的一条线。

| Aspect | Training | Inference |
|---|---|---|
| 目标 | 学参数 | 用参数生成 token |
| 输入 | 一整段 token sequence | prompt + 已生成 tokens |
| forward | 一次处理整段 sequence | 每步生成一个 token |
| loss | 计算 cross entropy | 通常不计算 training loss |
| backward | 需要 | 不需要 |
| optimizer step | 需要 | 不需要 |
| memory | activations + gradients + optimizer states | weights + KV cache |
| 关键优化 | throughput | latency / serving throughput |

训练阶段 expensive 的原因是除了 forward，还要保存 activations、做 backward、维护 optimizer states。推理阶段 expensive 的原因主要来自 repeated decoding、KV cache、memory bandwidth 和 batch serving。

## 9. Scaling And Compute

这组笔记回答的是：模型变大、数据变多、compute 变多时，loss 和成本怎么变化。

核心链条：

- [[Scaling Law]]：loss 随 model size、training tokens、compute 增加而下降。
- [[Training Compute - 6ND]]：dense Transformer 训练 compute 的粗略估算。
- [[FLOPs]]：衡量计算量。
- [[Model FLOPs Utilization]]：实际训练吞吐接近硬件峰值的程度。
- [[Resource Accounting]]：把 parameters、activations、gradients、optimizer states、KV cache、FLOPs 分开算。

常用粗略公式：

```math
C \approx 6ND
```

其中 $N$ 是 dense model parameters，$D$ 是 training tokens。

注意：$6ND$ 是粗略估算，忽略了 attention sequence length cost、embedding、optimizer overhead、communication overhead 等细节。

## 10. GPU Systems 主线

你的 GPU notes 其实在回答一个大问题：

> 为什么 Transformer workload 能在 GPU 上高效跑，以及瓶颈在哪里？

可以分成四层：

### 10.1 GPU execution model

- [[GPU]]：大量 threads 并行执行。
- [[SIMT]]：single instruction, multiple threads。
- [[GPU vs. CPU]]：CPU 偏 latency，GPU 偏 throughput。
- [[GPU Occupancy]]：SM 上 active warps 是否足够多。

### 10.2 Memory movement

- [[GPU Memory Bound]]：搬很多 bytes、做很少 FLOPs 的算子会被 memory bandwidth 限制。
- [[Memory Coalescing]]：连续访问让 memory transaction 更高效。
- [[Bank Conflict]]：shared memory 中多个 threads 访问同一 bank 会冲突。

### 10.3 Kernel optimization

- [[Tiling]]：把大矩阵分块，复用 shared memory / registers 中的数据。
- [[Operator fusion]]：把多个算子合并，减少 intermediate memory write/read。
- [[Recomputation]]：用额外 compute 换 activation memory。
- [[Low precision computation]]：减少 memory traffic，提高 tensor core throughput。

### 10.4 Performance diagnosis

- [[Arithmetic Intensity]]：FLOPs / bytes，判断偏 compute-bound 还是 memory-bound。
- [[GPU Bottleneck]]：定位瓶颈是 compute、memory、occupancy 还是 communication。
- [[Model FLOPs Utilization]]：衡量训练系统整体把 GPU 算力用起来了多少。

核心直觉：

> LLM systems optimization 很多时候不是“少算一点”，而是“少搬数据、提高数据复用、让 GPU 一直有活干”。

## 11. Advanced Architecture: MoE

[[Mixture of Experts (MoE)]] 这组笔记是 architecture 和 systems 交叉点。

MoE 的核心思想是：

- 模型总参数量很大；
- 但每个 token 只激活一小部分 experts；
- 目标是在增加 capacity 的同时控制 per-token compute。

主要问题：

- routing 是离散 top-k selection；
- expert usage 容易 imbalance；
- rich-get-richer 导致 expert collapse；
- device-level imbalance 会造成 straggler；
- all-to-all communication 变成 systems bottleneck。

[[MoE imbalance mitigation]] 的主线是：

> 实践中通常不会求解完整 routing / bandit 问题，而是在 normal language modeling loss 之外加入 load balancing loss 或 per-expert bias，让 experts 不要使用得太不均匀。

## 12. 推荐复习顺序

如果你想系统复习，可以按下面顺序走：

1. [[Language Modeling]]
2. [[Next-token prediction]]
3. [[Cross Entropy Loss]]
4. [[Tokenization]]
5. [[Transformer]]
6. [[Decoder-Only Transformer]]
7. [[Forward Pass in Transformer]]
8. [[Transformer Block]]
9. [[Self-Attention]]
10. [[Causal Mask]]
11. [[Residual Stream]]
12. [[RMSNorm]]
13. [[Rotary Position Embedding]]
14. [[Llama-style Architecture]]
15. [[Training Recipe]]
16. [[Training vs Inference]]
17. [[Scaling Law]]
18. [[Resource Accounting]]
19. [[Model FLOPs Utilization]]
20. [[GPU]]
21. [[Tiling]]
22. [[Flash Attention]]
23. [[Mixture of Experts (MoE)]]

## 13. 目前笔记的强项

- 你已经抓住了最核心的 conceptual distinction：objective、architecture、training recipe、systems 不是一回事。
- Transformer block 的理解很成熟，尤其是 residual stream、attention branch、MLP branch 的分工。
- 你在从 “model 是什么” 过渡到 “model 为什么贵、怎么跑得快”，这很接近 CS336 / LLM systems 的主线。
- 你没有只停留在公式，而是在反复写 intuition，这是很好的学习方式。

## 14. 可以补的缺口

下面这些链接在你的体系里已经频繁出现，但可以单独补成更完整的笔记：

- [[Language Modeling Head]]：hidden states 如何映射到 vocabulary logits，weight tying 是什么。
- [[Logits]]：为什么 logits 不是 probability，temperature 如何影响 distribution。
- [[KV Cache]]：推理时为什么要 cache K/V，memory cost 如何随 batch、layers、heads、context length 变化。
- [[Grouped Query Attention]]：为什么 GQA 可以降低 KV cache cost。
- [[Systems for Language Models]]：可以作为 systems 总入口，连接 GPU、MFU、Resource Accounting、Training vs Inference。
- [[AdamW]]：为什么 decoupled weight decay 对 LLM training 常见。
- [[Learning Rate Schedule]]：warmup + cosine decay 的直觉。
- [[Data Mixture]]：不同数据源比例如何影响 LM 行为。
- [[Evaluation]] / [[Perplexity]]：language model 怎么被评估。

## 15. 最后的整体理解

Language Modeling 可以看成一条完整工程链：

```text
用 next-token prediction 定义目标
-> 用 decoder-only Transformer 参数化条件分布
-> 用 cross entropy / likelihood 训练参数
-> 用 training recipe 决定优化路径
-> 用 scaling law 规划 model size 和 token count
-> 用 resource accounting 估算成本
-> 用 GPU systems optimization 提高实际吞吐
-> 用 inference optimization 支撑真实生成
```

一句话总结：

> 现代 LLM 不是单独一个 Transformer 概念，而是 objective、architecture、training recipe、data、scaling 和 systems 共同构成的一整套技术栈。

