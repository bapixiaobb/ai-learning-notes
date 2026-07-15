#AI #LanguageModeling #GPU

>**Note** — What is Memory Bound
>如果一个算子在 [GPU](<./GPU.md>) 上：搬很多 bytes，但只做很少 [FLOPs](<./FLOPs.md>)，那它就是 memory-bound。

典型例子：
```
elementwise add
layernorm
softmax
dropout
masking
```

这些操作计算量不大，但要读写大量 tensor。

# Memory-bound 对优化意味着什么？

1. 减少 HBM 读写
2. 提高 data reuse
	- 🔑：读一次，用很多次 $\rightarrow$ 提高 [Arithmetic Intensity](<./Arithmetic%20Intensity.md>)
3. 用 [tiling](<./tiling.md>) 把数据搬到快 memory
	- HBM → shared memory / local memory → register，先搬一个 tile 进来，然后在片上反复用。

>**Important**
>核心就是：减少 data movement

---
# 一些减少 Memory bound 的优化手段

- Trick 1: [Low precision computation](<../02%20-%20Training%20and%20Scaling/Low%20precision%20computation.md>)
- Trick 2: [Operator fusion](<./Operator%20fusion.md>)
- Trick 3: [Recomputation](<../02%20-%20Training%20and%20Scaling/Recomputation.md>)
- Trick 4: [Memory Coalescing](<./Memory%20Coalescing.md>) and DRAM
	- DRAM 不是你要一个数，它就只拿一个数。  它通常会按一整段连续 memory burst 取数据。所以如果你的线程刚好都要这一整段里的数据，那就赚了
	- 坏的情况：跳着访问，比如需要的数据都在不同的 DRAM block 里，每个 thread 都可能触发不同的 memory transaction，这就很浪费 bandwidth
- Trick 5 : [Tiling](<./Tiling.md>)

---

# What is wave quantization

这个名字容易和 [Quantization](<../../Transformer/Quantization.md>) 混，不是低精度量化。

这里的 **wave quantization** 指的是：

> tile 数量和 SM 数量不匹配，导致最后一波只剩很少 tile，很多 SM 空闲。

>**Example** — 比如 A100 有 108 个 SM
>如果你有：98 个 tiles
>>那一次 wave 里最多只有 98 个 SM 有活干，剩下一些 SM 空闲，但还算可接受。
>
>如果你变成：120 个 tiles
>>那么第一波 108 个 tile 跑完后，还剩：12 个 tiles
>>第二波只有 12 个 SM 有活干，其他 96 个 SM 空闲
>
>所以性能可能突然掉。

这就是为什么：

> 矩阵维度只加了 1，吞吐却大幅下降。

因为 tile decomposition 发生了离散跳变。
