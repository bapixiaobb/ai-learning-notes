#AI #GPU

在 [Parallelism](Parallelism.md) 中，有一个比较重要的 [NCCL](NVIDIA%20Collective%20Communication%20Library.md) 的等价关系

```
all-reduce = reduce-scatter + all-gather
```

![all-reduce = reduce-scatter + all-gather](all-reduce%20%3D%20reduce-scatter%20%2B%20all-gather.png)

---
> **Example:** 每个 rank 都创建一个 tensor：
>```
>rank 0: [0, 1, 2, 3]
>rank 1: [1, 2, 3, 4]
>rank 2: [2, 3, 4, 5]
>rank 3: [3, 4, 5, 6]
>```
而且它们分别放在各自 GPU 上：
```
rank 0 的 data 在 cuda:0
rank 1 的 data 在 cuda:1
rank 2 的 data 在 cuda:2
rank 3 的 data 在 cuda:3
```
## All-reduce
`all_reduce` 做两件事：
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

## Reduce-scatter
`reduce_scatter` 也是两步：
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

## All-gather
这里把刚才 reduce-scatter 得到的小片段作为新的 input：
```
rank 0 input = [6]
rank 1 input = [10]
rank 2 input = [14]
rank 3 input = [18]
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
