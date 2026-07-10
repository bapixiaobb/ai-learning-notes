#LanguageModeling #Systems #FLOPs

>[!note]
> **Resource Accounting** 是在训练 large language model 前，对计算量、显存、内存带宽、训练时间和硬件利用率进行估算的过程。
>
> 它的目的不是单纯“数 FLOPs”，而是判断一个模型训练计划在给定硬件和预算下是否可行，以及瓶颈可能在哪里。

## 为什么需要 Resource Accounting

>[!note]
> 大模型训练成本极高，不能先随便跑再看结果。训练前必须估算：
>
> - 需要多少 total compute
> - 需要多少 [[GPU]] hours
> - 显存是否放得下
> - training throughput 是否足够
> - bottleneck 是 compute 还是 [[GPU Memory Bound]]
> - scaling forecast 是否合理

对于 language model，训练成本通常由几个量共同决定：

$$
\text{model size}
,\quad
\text{number of training tokens}
,\quad
\text{sequence length}
,\quad
\text{hardware throughput}
,\quad
\text{memory constraints}
$$

>[!question]
> 给定模型规模、训练数据和硬件，我们能不能在预算内完成训练？如果不能，瓶颈在哪里？

## 核心估算对象

### Compute

>[!note]
> Compute 衡量训练总共需要多少 floating-point operations。它通常用 [[FLOPs]] 表示。

对于 dense Transformer training，常用粗略估算是：

$$
\text{Training FLOPs} \approx 6ND
$$

其中：

- $N$：number of model parameters
- $D$：number of training tokens

>[!note]
> 直觉是：每个 token 训练时都要经过整个 dense model；单个 token 的计算量大约正比于参数量 $N$，训练 $D$ 个 tokens 后总计算量约正比于 $ND$。

### Throughput

>[!note]
> Throughput 衡量训练系统实际跑得多快，常见单位包括 FLOP/s、tokens/s 或 samples/s。

### Memory

>[!note]
> Memory 关注训练时显存是否放得下。大模型训练中，显存通常被 parameters、gradients、optimizer states 和 activations 占用。

主要显存来源：

| 部分 | 含义 |
|---|---|
| Parameters | 模型权重 |
| Gradients | 反向传播得到的梯度 |
| Optimizer states | Adam 等 optimizer 保存的动量和二阶矩 |
| Activations | forward pass 中为 backward 保存的中间结果 |

>[!note]
> 很多时候模型不是 FLOPs 不够跑，而是显存放不下，或者 memory bandwidth 供应不上计算单元。

### Bandwidth

>[!note]
> Memory bandwidth 衡量单位时间内能从 memory 搬运多少数据。
>
> 在 accelerator 上，很多操作的瓶颈不是计算，而是数据搬运。


## 判断 Bottleneck

>[!note] [[GPU Bottleneck]] 
> Resource accounting 不只是估算“需要多少计算”，还要判断训练过程是 compute-bound 还是 [[GPU Memory Bound]]。
## 重要 Techniques

### Gradient Accumulation

>[!note]
> [[Gradient Accumulation]] 用于在显存放不下 large batch 时，用多个 microbatches 累加梯度，再做一次 optimizer step。


### Activation Checkpointing

>[!note]
> Activation Checkpointing 用额外 recomputation 换取更低 activation memory。
>
> forward 时不保存所有 intermediate activations；backward 时需要时再重新计算。

核心 trade-off：

$$
\text{memory} \downarrow,
\quad
\text{compute} \uparrow
$$

>[!note]
> 它常用于深层 Transformer 或长 sequence 训练，因为 activations 会随 batch size、sequence length 和 layer 数增长。

### Mixed Precision

>[!note]
> Mixed precision 使用低精度数据类型，例如 FP16、BF16 或 FP8，减少 memory usage，提高 accelerator throughput。

它可以降低：

- parameter memory
- activation memory
- memory bandwidth pressure

同时提高：

- Tensor Core utilization
- training throughput

>[!note]
> Mixed precision 需要注意 numerical stability，因此通常会结合 loss scaling、BF16 accumulation 或特定 optimizer 设置。

### Parallelism

>[!note]
> 当单个设备无法容纳或高效训练模型时，需要使用 parallelism 将训练分布到多个 devices 上。

常见方式包括：

- data parallelism
- tensor parallelism
- pipeline parallelism
- sequence parallelism
- expert parallelism

>[!note]
> Parallelism 可以扩展训练规模，但也会引入 communication overhead。因此 resource accounting 需要同时考虑 computation 和 communication。

### Model FLOPs Utilization

>[!note]
> [[Model FLOPs Utilization]] 衡量模型实际训练时使用了多少硬件理论峰值算力。

形式上可以理解为：

$$
\text{MFU}
=
\frac{\text{model FLOP/s}}{\text{hardware peak FLOP/s}}
$$

>[!note]
> 高 FLOPs 估算并不代表训练效率高。实际系统还可能被 memory bandwidth、communication、kernel efficiency 或 batch size 限制。

## 和 Scaling Law 的关系

>[!note]
> Resource accounting 还服务于 scaling decision：给定 compute budget，应该选择多大的模型、多少训练 tokens、什么 batch size 和训练时长。

典型问题是：

>[!question]
> 如果预算是固定的，是应该训练更大的模型，还是用更多 tokens 训练较小的模型？

这类问题和 [[Scaling Law]]、Compute Optimal Training 直接相关。

## 总结

>[!note]
> Resource accounting 是 large language model training 前的系统性估算过程。它连接了模型规模、数据规模、FLOPs、memory、bandwidth、parallelism 和 training cost。
>
> 它的核心作用是：在训练前判断计划是否可行，在训练中定位 bottleneck，并指导优化方向。

## Related

- [[Language Modeling]]
- [[Transformer]]
- [[FLOPs]] 
- [[Training Compute - 6ND]]
- [[Model FLOPs Utilization]]
- [[Scaling Law]] 