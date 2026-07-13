#AI #LanguageModeling #GPU

NCCL 是 NVIDIA 的 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) communication library。

它不决定 [Parallelism](<./Parallelism.md>) 怎么切；它负责把 collective operations 高效地跑在底层硬件上，比如 NVLink / NVSwitch / InfiniBand。

可以理解成：

```
Parallelism strategy 决定需要什么通信
        ↓
collective operation 表达通信模式
        ↓
NCCL 负责具体怎么传
```

# What NCCL does

- 识别 [GPU Communication Topology](<./GPU%20Communication%20Topology.md>)，比如哪些 GPU 在同一个 node、哪些路径走 NVLink / NVSwitch。
- 为 collective operation 选择通信路径。
- launch GPU kernels 来 send / receive data。

# Examples

## Data parallelism

每个 rank 有完整 model，但看到不同 batch shard。  
backward 后 gradients 不同，所以需要：

```python
dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
```

NCCL 负责把所有 GPU 上的 gradients 平均，并让每个 rank 拿到同一个结果。

## Tensor parallelism

每个 rank 只算出 activation 的一部分。  
如果下一步需要完整 activation，就需要：

```python
dist.all_gather(...)
```

NCCL 负责把各个 rank 的 activation shard 收集起来，让每个 rank 都拿到完整 tensor。

## FSDP / ZeRO intuition

如果 parameters / gradients 被 shard 到不同 rank 上，常见 pattern 是：

```text
all-gather parameters for forward
reduce-scatter gradients after backward
```

[All-reduce Decomposition](<./All-reduce%20Decomposition.md>)

NCCL 负责执行这些通信；显存节省来自 sharding strategy，不是 NCCL 本身。

---
# 🔗
[Parallelism](<./Parallelism.md>)
[GPU Communication Topology](<./GPU%20Communication%20Topology.md>)
