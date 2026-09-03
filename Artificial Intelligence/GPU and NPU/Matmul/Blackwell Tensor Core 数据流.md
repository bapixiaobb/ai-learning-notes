#GPU

```
GMEM
  │ TMA load
  ▼
Shared Memory Pipeline：A/B
  │ SMEM descriptors
  │ 一个 Thread 发起 tcgen05.mma
  ▼
5th Gen Tensor Core
  │
  ▼
TMEM accumulator
  │ tcgen05.ld
  ▼
Epilogue Warp 的 Registers
  │ CUDA Core 后处理
  ▼
Shared Memory
  │ TMA store
  ▼
GMEM
```

## TMEM (Tensor Memory)

TMEM 是第五代 Tensor Core 专用的片上存储，不属于普通 Thread Register，也不等于 Shared Memory。

Blackwell 把矩阵累加结果从 `Thread private registers` 移到了 `Tensor Memory`

⚠️ 这里还是用了 Thread Registers，但是不一样的是，[H100](<H100%20Tensor%20Core%20%E6%95%B0%E6%8D%AE%E6%B5%81.md>)的 accumulator 是放在 Registers 上的，但是这里 accumulator 始终放在 TMEM 上，等到搬运的时候，再放到 Register 上

## `tcgen05.ld`和 WGMMA 有什么不同？
### H100 WGMMA

```
4个 Warp = 128个 Thread
          │
          │ 所有线程共同执行 WGMMA
          ▼
       Tensor Core
          │
          ▼
结果分散在这128个 Thread 的 registers
```

### Blackwell `tcgen05.mma`

```
一个 Thread
    │ 发起 tcgen05.mma
    ▼
Tensor Core 执行整个矩阵 MMA
    │
    ▼
结果写入 TMEM
```

## A、B、D 分别放在哪里？

常见 Blackwell GEMM 路径中：

```
A：SMEM，也可以在部分模式下来自 TMEM
B：SMEM
D/accumulator：一定在 TMEM
```

对应指令可以抽象成：

```
tcgen05.mma(
    destination = TMEM,
    A = SMEM descriptor,
    B = SMEM descriptor
)
```

所以它不再是：

```
mma.sync {d_registers}, {a_registers}, {b_registers}
```

而更像：

```
tcgen05.mma [d_tmem], a_descriptor, b_descriptor
```

这里的 `a_descriptor`、`b_descriptor` 只是描述 SMEM 中矩阵的位置和布局，不承载整个矩阵数据。

## 最终结果怎么离开 TMEM？

TMEM 不是 CUDA Core 普通标量/向量指令可以随便读取的 register。需要通过专门的 `tcgen05.ld`：

```
TMEM accumulator
  │ tcgen05.ld
  ▼
Thread Registers
  │ CUDA Core 做激活、缩放、类型转换等 epilogue
  ▼
SMEM
  │ TMA store
  ▼
GMEM
```

所以 Blackwell 不是彻底消灭了结果侧寄存器，而是改变了寄存器的占用时间：

```
H100：
整个 MMA 累加过程，结果始终占用 registers

Blackwell：
累加过程留在 TMEM
只有真正进行 epilogue 时，才把需要的结果加载到 registers
```

这会显著减少长期 accumulator register 压力，并允许另一批 Warp 接手 epilogue。

## 异步

```
时间 ─────────────────────────────────────→

TMA:       Load Tile2     Load Tile3
Tensor Core:    MMA Tile1      MMA Tile2
Epilogue:            Result0       Result1
```
