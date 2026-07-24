#AI #LanguageModeling #GPU

是 [GPU Memory Bound](<./GPU%20Memory%20Bound.md>) 的一种重要的针对 Matmul 的优化方法

>**Important** — Intuition
>Tiling = 把大矩阵乘法拆成小块，让每一小块数据搬进快 memory 后，被尽可能多次复用，再写回慢 memory。

它不是单纯“切块”，而是适配 [GPU](<./GPU.md>) 类似硬件的方法：
```
切块
↓
搬进 shared/local memory
↓
反复用
↓
累加 partial result
↓
最后写回 global memory
```
---
# Basic Matmul
假设：
```math
C=AB
```
其中：

```
A: M × K
B: K × N
C: M × N
```

一个输出元素是：
```math
C_{ij} = \sum_k A_{ik} B_{kj}
```

朴素理解是：

```
算 C[i, j]
就去读 A 的第 i 行
再读 B 的第 j 列
然后点乘
```

问题是：**同一个 A 元素、B 元素会被反复从 global memory 读取。** 如下图所示
![Matrix Multiplication.png](<../../attachments/Matrix%20Multiplication.png>)
$M_{0,0}$ 会被 output 的首行每个元素的计算都用到，所以 matmul 天然有大量 **data reuse**。

---
# ❓ What problem dose Tiling solve？

>**Question** — 既然同一批 A/B 数据会被反复用，为什么每次都从 global memory 读？

Tiling 的想法是：

```
先把一小块 A 和一小块 B 搬到快 memory
然后在快 memory 里反复用它们
算完一个 C 的小块
最后再写回 global memory
```
---

# 🧮 How to compute a C Tile？

一个 M tile 和一个 N tile 的组合，可以一次更新一个 P tile：

![Tiling MatMul.png](<../../attachments/Tiling%20MatMul.png>)
M tile 读进来一次，N tile 读进来一次，在 local/shared/register 里反复用，一次性算出多个 P 元素

---

>**Question** — 为什么 tiling 会快？
>因为它把 global memory access 换成了 fast memory access。

没有 tiling 的情况下：每次需要 A/B都去 global memory 读
有 tiling 的情况下：A_tile / B_tile 从 global memory 读一次之后在 shared/local/register 里反复用

>**Note** — Tiling Math
>如果 tile size 是 T，那么 global memory access 大约可以减少 T 倍。
>因为原来每个元素可能要从 global memory 读很多次。
>现在变成：global memory 读少量次数shared/local memory 读很多次
>这就把瓶颈从慢 memory 挪到了快 memory。

---

# 🪡 Sizing Tile？

要平衡几件事：
### ① Tile 太小
问题是：每次搬进来的数据复用不够`

虽然每个 tile 很轻，但 data reuse 不强。

结果是：
- global memory traffic 还是偏多
- arithmetic intensity 不够高
- tensor core 可能吃不满

也就是：**tile 太小，复用不够。**
### ② Tile 太大

问题是：
- shared/local memory 放不下
- register pressure 太高
- occupancy 下降
- 并行 block 数变少

也就是说：单个 tile 很爽，但同时跑的 tile 数变少，SM / AI Core 可能闲着

也就是：**tile 太大，占资源太多。**
### ③ Tile 不对齐

比如矩阵维度不是 tile size 的整数倍：
那就会出现：最后一个 tile 可能skinny，存在大量的 empty space

这个 tile 很尴尬：硬件还是要调度它但里面有效计算很少

>**Note**
> 所以很多时候 padding 会变快。
> 比如把：257 padding 到 288。看起来多算了一点，但因为 tile 更规整、memory 更对齐、coalescing 更好，反而快。

>**Question** — tile size 经常和 16、32、64、128 有关？
>是因为硬件有自己的粒度：
>```
>warp size
>memory transaction / burst size
>tensor core MMA shape
>cache line / bank layout
>alignment requirement
>```
>所以如果你的矩阵维度能被：16 / 32 / 64 / 128 整除，通常更容易：
>>[Memory Coalescing](<./Memory%20Coalescing.md>)
>>vectorized load/store
>>避免 ragged tile
>>吃满 tensor core
>>减少 boundary handling。
>
>这就是为什么很多模型配置喜欢：
>
>```
>hidden_dim = 4096
>intermediate_dim = 11008
>vocab padded to multiple of 64/128
>```
>不是审美问题，是系统性能问题。⚙️

---

# Tiling 和 coalescing 的关系

Tiling 不是孤立的。你可以选一个看起来很合理的 tile：128 × 128。但如果它在 memory 里是错位的，读取时可能不能 coalesce。

比如你想读一个二维 tile：一行一行连续读，如果每一行刚好对齐 memory burst window，读取很舒服。但如果矩阵宽度偏了 1，下一行起始地址就错位了。![Memory Alignment.png](<../../attachments/Memory%20Alignment.png>)

结果可能变成：本来一次 burst 能读完现在要两次 burst，所以 tile size、矩阵 shape、layout、padding 会一起影响性能。

>**Note** — 所以调 tiling 经常是玄学感很强：改一个 shape性能突然掉很多
>它背后通常不是数学复杂度变了，而是：
>>tile 数量变了
>>alignment 变了
>>coalescing 变了
>>occupancy 变了
>

