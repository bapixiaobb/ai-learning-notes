#AI #LanguageModeling #GPU

All-Gather 适合这样的情况：每个 rank 只保存一个 local shard，但接下来的计算需要完整 tensor，于是互相交换 shards，让所有 ranks 都临时重建出 global tensor。

这使它自然连接到 [ZeRO](<./ZeRO.md>) / FSDP 和 [Tensor Parallelism](<./Tensor%20Parallelism.md>)：平时通过 sharding 节省 memory，需要完整 parameters 或 [activations](<../../Neural%20Networks/Activations.md>) 时再 All-Gather。它也是 [All-reduce Decomposition](<./All-reduce%20Decomposition.md>) 的后半段。

这个需求由 [Parallelism](<./Parallelism.md>) 决定，实际传输通常由 [NCCL](<./NVIDIA%20Collective%20Communication%20Library.md>) 执行。

![All-Gather.png](<../attachments/All-Gather.png>)
## Example

每个 rank 都创建一个 tensor：
```
rank 0: [0, 1, 2, 3]
rank 1: [1, 2, 3, 4]
rank 2: [2, 3, 4, 5]
rank 3: [3, 4, 5, 6]
```

而且它们分别放在各自 GPU 上：
```
rank 0 的 data 在 cuda:0
rank 1 的 data 在 cuda:1
rank 2 的 data 在 cuda:2
rank 3 的 data 在 cuda:3
```

为了和 [All-reduce Decomposition](<./All-reduce%20Decomposition.md>) 共用同一个例子，这里把 [Reduce-Scatter](<./Reduce-Scatter.md>) 的输出当作各 rank 的 local shards。All-Gather 本身不负责 reduce，也不要求前一步必须是 Reduce-Scatter。

```
rank 0 output = [6]
rank 1 output = [10]
rank 2 output = [14]
rank 3 output = [18]
```

`all_gather` 做的是：

> 每个 rank 都把自己的小片段贡献出来，然后所有 rank 都拿到拼好的完整结果。

所以执行完之后，每个 rank 都有：

```
rank 0: [6, 10, 14, 18]
rank 1: [6, 10, 14, 18]
rank 2: [6, 10, 14, 18]
rank 3: [6, 10, 14, 18]
```

>**Summary**
> All-Gather 的输出是 **replicated global tensor**：输入是分散的 shards，输出是每个 rank 都拥有的完整拼接结果。
