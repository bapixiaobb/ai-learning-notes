#AI #LanguageModeling

All-Reduce 适合这样的情况：每个 rank 都算出了一个 local tensor，需要先把它们聚合成 global result，而且之后所有 ranks 都需要这份完整结果。

在 [Data parallelism](<./Data%20parallelism.md>) 中，每个 rank 根据自己的 local batch 得到 local gradients；All-Reduce 把它们求和或平均，让所有 ranks 在 optimizer update 前重新拿到一致的 global gradients。

这是 [Parallelism](<./Parallelism.md>) 决定“需要同步什么”之后产生的 communication pattern；[NCCL](<./NVIDIA%20Collective%20Communication%20Library.md>) 负责让它在实际 GPU topology 上高效执行。它也可以分解为 [Reduce-Scatter + All-Gather](<./All-reduce%20Decomposition.md>)。

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

all_reduce 做两件事：

1. 把所有 rank 的 tensor 按位置加起来；
2. 把加完的完整结果发回给所有 rank。

```
rank0: [0, 1, 2, 3]
rank1: [1, 2, 3, 4]
rank2: [2, 3, 4, 5]
rank3: [3, 4, 5, 6]
--------------------
sum:   [6, 10, 14, 18]
```

所以 all-reduce 之后，每个 rank 的 `data` 都变成：

```
rank 0: [6, 10, 14, 18]
rank 1: [6, 10, 14, 18]
rank 2: [6, 10, 14, 18]
rank 3: [6, 10, 14, 18]
```
