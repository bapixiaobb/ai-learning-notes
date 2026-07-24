#AI #LanguageModeling

Reduce-Scatter 适合这样的情况：多个 ranks 各自持有 partial values，需要先做 reduce，但 reduce 后的完整结果不必在每个 rank 上复制，只需要让每个 rank 保留其中一个 shard。

这正好连接到 [ZeRO](<./ZeRO.md>)：gradients 聚合后可以继续保持 sharded，每个 rank 只负责自己那一部分 parameters / optimizer states。与 [All-Gather](<./All-Gather.md>) 连起来时，它构成 [All-reduce Decomposition](<./All-reduce%20Decomposition.md>) 的前半段。

这里的 communication pattern 来自 [Parallelism](<./Parallelism.md>) 的 sharding strategy，实际传输通常由 [NCCL](<./NVIDIA%20Collective%20Communication%20Library.md>) 执行。

![reduce-scatter.png](<../../attachments/reduce-scatter.png>)
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

`reduce_scatter` 两步：
1. 先 reduce，也就是逐元素求和；
2. 再 scatter，把结果切开分给不同 rank。

先求和：
```
[6, 10, 14, 18]
```

再 scatter：

```
rank 0 gets [6]
rank 1 gets [10]
rank 2 gets [14]
rank 3 gets [18]
```

所以 reduce-scatter 之后：

```
rank 0 output = [6]
rank 1 output = [10]
rank 2 output = [14]
rank 3 output = [18]
```

>**Summary**
> Reduce-Scatter 的输出是 **reduced shards**：信息已经在全局上完成聚合，但结果仍分散在不同 ranks 上。
