#GPU

```
GMEM
  │ TMA
  ▼
Shared Memory Pipeline
  │ Shared Memory descriptor
  │ wgmma.mma_async
  ▼
Tensor Core
  │
  ▼
Warpgroup accumulator registers
  │ epilogue
  ▼
Shared Memory
  │ TMA store
  ▼
GMEM
```

## TMA (Tensor Memory Accelerator)

[A100 的 cp.async](<A100%20Tensor%20Core%20%E6%95%B0%E6%8D%AE%E6%B5%81.md#async-的意义>) 假设搬运一个 Tile
```
Warp 中很多 Thread
    │
    ├─ cp.async 一小块
    ├─ cp.async 一小块
    ├─ cp.async 一小块
    └─ ...
            ↓
          SMEM
```
虽然不经过寄存器，但仍需要：
- 多条搬运指令；
- 多个 Thread 计算各自的地址；
- Warp Scheduler 发射这些指令。

#### TMA 的做法是
```
一个选定的 Thread
        │
        │ 提交一条“大块、多维 Tensor 搬运命令”
        ▼
       TMA
        │ 硬件负责地址生成和实际搬运
        ▼
      SMEM Tile
```

这里“一个 Thread 发起”不代表这个 Thread 自己逐字节搬运，而是它向 TMA 提交任务。TMA 可以描述一维到五维 Tensor，在 GMEM 和 SMEM 间异步搬运；完成后通过 barrier 通知使用者。[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/pdf/Hopper_Tuning_Guide.pdf)、[CUDA TMA 编程说明](https://docs.nvidia.com/cuda/cuda-program-guide/04-special-topics/async-copies.html#using-the-tensor-memory-accelerator-tma)

所以：

```
cp.async：很多小搬运指令，共同组成一个 Tile
TMA：一条命令描述一个大块、多维 Tile
```

这个很像 MTE2。

## WGMMA (Warpgroup Matrix Multiply-Accumulate)

A100 的 `mma.sync` 是一个 Warp，也就是32个 Thread 共同执行；H100 的 `wgmma.mma_async` 是一个 Warpgroup，也就是连续的4个 Warp、共128个 Thread 共同执行一个更大的矩阵操作。

```
A100:
1 Warp = 32 Threads
        │ mma.sync
        ▼
    Tensor Core

H100:
1 Warpgroup = 4 Warps = 128 Threads
        │ wgmma.mma_async
        ▼
      Tensor Core
```

更重要的是，WGMMA 支持用 Shared Memory descriptor 描述输入矩阵：

```
A100 常见路径：

SMEM
  │ ldmatrix
  ▼
A/B register fragments
  │ mma.sync
  ▼
Tensor Core
```

```
H100 常见 WGMMA 路径：

SMEM 中的 A/B
  │ Shared Memory descriptor
  │ wgmma.mma_async
  ▼
Tensor Core
```


但要注意一个精确边界：

- WGMMA 的 B 必须位于 Shared Memory；
- A 可以位于 Shared Memory，也可以仍然使用 register fragment；
- 计算结果 accumulator 仍然位于 Warpgroup 的分布式普通寄存器中。

因此不能笼统地说“H100 消灭了所有 A/B 寄存器”，只能说：

> 在 A、B 都使用 Shared Memory operand 的 WGMMA 路径中，H100 消除了显式 `ldmatrix → A/B register fragments` 这段程序可见中转。

这些操作数约束由 NVIDIA 的 [PTX WGMMA 文档](https://docs.nvidia.com/cuda/archive/12.5.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)明确规定。

## Warp specialization

H100 还可以把不同 Warp 固定分配给不同工作：

```
Producer Warp：
TMA 搬入 Tile 1、Tile 2、Tile 3……

Consumer Warpgroup：
WGMMA 计算 Tile 0、Tile 1、Tile 2……
```

时间上：

```
时间 ─────────────────────────────────→

Producer:  TMA Tile1    TMA Tile2    TMA Tile3
Consumer:       WGMMA Tile0   WGMMA Tile1   WGMMA Tile2
```

```
A100：
同一批 Warp 在指令流中交替提交 copy / compute

H100 常见优化：
不同 Warp 分别作为 producer / consumer
```

同步关系仍然和 Ascend ping-pong 一样
