#AI #LanguageModeling #GPU
# Width-wise parallel

We can think of [pipeline parallel](<./Pipeline%20Parallelism.md>) as cutting up along depth. What about width?

Tensor Parallelism 是沿 hidden dimension (h) 切，具体 shape 见 [Transformer](<../../Transformer/Transformer.md>) 最后的那张图
![Decomposition of matrix multiplication.png](<../../attachments/Decomposition%20of%20matrix%20multiplication.png>)

图中：

```math
X=[X_1\;X_2],
\qquad
A=
\begin{bmatrix}
A_1\\
A_2
\end{bmatrix}
```

因此：

```math
Y=XA=X_1A_1+X_2A_2
```

每个 rank 分别计算一个 partial output：

```math
Y_1=X_1A_1,\qquad Y_2=X_2A_2
```

最后通过 [All-Reduce](<./All-Reduce.md>) 求和，恢复完整的 $Y$。

## Trade-off

Tensor Parallelism 把同一个 layer 的 parameters、activations 和 matrix multiplication 分到多个 ranks，因此不需要等待不同 layers 依次执行，也没有 pipeline bubble。

代价是每一层附近都可能需要 activation-sized collective communication。所以它依赖高bandwidth、低 latency 的连接，通常优先放在同一 node 内通过 NVLink / NVSwitch 连接的 GPUs 上。

相比之下，[Pipeline Parallelism](<./Pipeline%20Parallelism.md>) 通信频率较低且主要是相邻 stages 之间的 point-to-point communication，因此更适合跨 node 的慢链路。

# 概念类似于 [Tiling](<../03%20-%20GPU%20and%20Systems/Tiling.md>)

|                    | Tiling                                  | Tensor Parallelism                        |
| ------------------ | --------------------------------------- | ----------------------------------------- |
| 分块发生在哪里            | 一张 GPU 内部                               | 多张 GPU 之间                                 |
| 主要目的               | 利用 shared memory/register，提高 data reuse | 分摊 parameters、compute 和 activation memory |
| 数据放在哪里             | 完整矩阵通常在同一张卡的 HBM，临时加载 tiles             | 每张卡长期只持有 matrix shard                     |
| 合并 partial results | GPU 内部累加                                | 跨 GPU collective，例如 All-Reduce            |
| 通信路径               | HBM ↔ shared memory ↔ registers         | GPU ↔ GPU，NVLink / network                |

 tensor parallel 和 tiling 代数完全相同，区别是 partial results 在哪里：
- Tiling：不同 tiles 仍在同一张 GPU 上，进行本地累加。
- Tensor Parallelism：不同 shards 位于不同 GPUs，必须通过通信完成累加。

而且它们实际上会嵌套使用：

```
完整 matrix multiplication
        ↓ Tensor Parallelism
分给不同 GPUs 的 matrix shards
        ↓ 每张 GPU 内部继续 Tiling
由 thread blocks / warps 执行 local matmul
```

因此最准确的一句话是：

> **Tiling 是单卡内部的矩阵分块执行；Tensor Parallelism 是跨卡的矩阵分片。Tensor Parallelism 的每个 rank 内部仍然会使用 tiling。**

