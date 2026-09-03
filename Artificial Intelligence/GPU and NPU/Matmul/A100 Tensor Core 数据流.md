#GPU

```
GMEM
  │ cp.async
  ▼
Shared Memory
  │ ldmatrix
  ▼
Warp 的分布式寄存器 fragments
  │ mma.sync
  ▼
Tensor Core
  │
  ▼
accumulator registers
```

Ampere 加入 `cp.async`：

```
GMEM
  ║
  ║ cp.async
  ▼
SMEM
```

从程序员可见的数据路径看，它不再需要：

```
GMEM → 临时 Thread Register → SMEM
```

而是：

```
GMEM ═══════════════> SMEM
          cp.async
```

注意：Ampere 只消掉了**第一处中转寄存器**。

下面两部分仍然存在：

```
SMEM → A/B register fragments → Tensor Core
Tensor Core → accumulator registers
```

所以不能理解成“Ampere Tensor Core 可以直接吃 Shared Memory”。

## `ldmatrix`

>**Note** — `ldmatrix` 让整个 Warp 协作读取 Shared Memory 中的矩阵，并把矩阵切片分发到各个 Thread 的寄存器中，供后续 `mma.sync` 使用。

如果不用 `ldmatrix`，理论上也可以让每个 Thread 使用普通 Shared Memory load，再自己计算地址、重新排列和交换数据。但这会产生大量细碎加载与布局处理。`ldmatrix` 把“Warp 协作加载矩阵并按 MMA 所需方式分发”变成了一条专用指令。[NVIDIA PTX ISA 对 `ldmatrix` 的定义](https://docs.nvidia.com/cuda/archive/11.0/parallel-thread-execution/index.html)

`ldmatrix` 不是 A100 才首次出现，它从 `sm_75`，也就是 Turing 时代就已经支持；A100/Ampere 把它和 `cp.async` 组合成了更高效的矩阵流水。
## `async` 的意义

`cp.async` 发出以后，Warp 不需要原地等待整个拷贝完成：

```
发起 cp.async：搬运 Tile 1
        │
        │ copy 在后台推进
        ▼
计算 Tile 0
```

于是可以做 ping-pong：

```
SMEM Buffer 0：Tensor Core 正在计算 Tile 0
SMEM Buffer 1：cp.async 正在搬入 Tile 1
```

时间上：

```
时间 ─────────────────────────────→

Copy:     Tile 0     Tile 1     Tile 2
Compute:        Tile 0     Tile 1     Tile 2
```

这和 [Ascend](<../Ascend%20NPU.md>) 流水思路完全一致：

- 不同 Tile 使用不同 Buffer；
- 搬运和计算重叠；
- Consumer 使用 Buffer 前必须等待数据 ready；
- Buffer 被下一轮覆盖前，必须确保上一轮 Consumer 已经使用完。

因此：

> `cp.async` 提供异步搬运能力，ping-pong buffer 解决存储冲突，wait/barrier 保证依赖正确。

三者不是一回事。

## [GPU](<../GPU.md>) 和 [NPU](<../Ascend%20NPU.md>) 越来越像了

`cp.async` 开始具有 MTE2 式的“搬运与计算解耦”味道，但它还是 Thread/Warp 发起的小粒度 copy；[Hopper](<H100%20Tensor%20Core%20%E6%95%B0%E6%8D%AE%E6%B5%81.md>) 的 TMA 才更接近“一条命令描述整个大块、多维搬运”。
