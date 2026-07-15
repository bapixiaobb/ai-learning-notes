#AI #GPU

[All-Reduce](<./All-Reduce.md>) 不一定要被看成一个不可拆分的操作。它可以分成：

```
all-reduce = reduce-scatter + all-gather
```

![all-reduce = reduce-scatter + all-gather.png](<../attachments/all-reduce%20%3D%20reduce-scatter%20%2B%20all-gather.png>) [All-Reduce](<./All-Reduce.md>) = [Reduce-Scatter](<./Reduce-Scatter.md>) + [All-Gather](<./All-Gather.md>)

[Reduce-Scatter](<./Reduce-Scatter.md>) 先完成聚合，但让每个 rank 只保留一个 reduced shard；[All-Gather](<./All-Gather.md>) 再交换这些 shards，让每个 rank 都得到完整结果。因此两步合起来，输出与 All-Reduce 相同。

这个等价关系把两类 [Parallelism](<./Parallelism.md>) 状态连接起来：

- naïve [Data parallelism](<./Data%20parallelism.md>) 需要每个 rank 都持有完整 gradients，所以表现为 All-Reduce；
- [ZeRO](<./ZeRO.md>) 希望 gradients / optimizer states 保持 sharded，因此可以在 Reduce-Scatter 后先处理各自的 shard，需要完整 parameters 时再 All-Gather。

>**Note**
> 在 bandwidth-limited 的实现中，优化后的 All-Reduce 本来通常就由 Reduce-Scatter + All-Gather 完成，因此 ZeRO-1 能在不增加主要通信量的情况下节省 optimizer-state memory。[NCCL](<./NVIDIA%20Collective%20Communication%20Library.md>) 负责把这些 collective operations 映射到实际通信拓扑。
