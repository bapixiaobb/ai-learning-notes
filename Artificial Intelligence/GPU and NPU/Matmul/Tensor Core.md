#GPU

[GPU](<../GPU.md>) 上专门计算矩阵 Tile 的矩阵执行单元

>**Note** — Volta 是 NVIDIA 第一次引入 Tensor Core 的架构。(V100)
>```
Volta     →     V100     →     第一次有 Tensor Core
  ↓
Ampere    →     A100     →     cp.async，减少搬运的寄存器中转
  ↓
Hopper    →     H100     →     TMA + WGMMA，异步流水和 warp specialization
  ↓
Blackwell →     B100 / B200     →     TMEM，矩阵累加结果进一步脱离普通寄存器
>```
## Why Tensor Core

Matrix Multiplication 最终可以展开为大量 scalar FMA。传统 [CUDA Core](<../CUDA%20Core.md>) 通过大量线程执行 scalar [FMA](<../../Fundamentals/Fused%20Multiply-Add.md>): 每个 [Thread](<../Thread.md>) 需要把自己使用的 A、B 数据装进私有寄存器，再反复执行 FMA。

传统 [CUDA Core](<../CUDA%20Core.md>) 在计算 $C=A\times B$ 的大致做法是：

```
GMEM
  ↓
Shared Memory：保存 A/B tile，供线程块复用
  ↓
每个 Thread 的私有 Registers
  ↓
CUDA Core 做 scalar FMA
  ↓
每个 Thread 的结果 Registers
```

因此它虽然已经利用 Shared Memory 做了线程块级复用，但仍然存在：

- 每个线程需要保存自己的输入和累加结果；
- 需要发射大量普通 FMA 指令；
- 不同线程间的矩阵数据复用表达得不够直接；
- 遇到访存延迟，主要靠切换到其他 ready warp 隐藏。

Tensor Core 则提供专用的 [MMA](<../Matrix%20Multiply-Accumulate.md>) 数据通路，让一个 Warp/Warp-group 协作完成矩阵 Tile 级的乘累加。

## Matrix Array and Data Reuse

- MatMul 中一个 A/B 元素会参与多个输出元素，因此存在天然的数据复用。
- 脉动阵列展示了一种解决办法：A/B 在 PE 阵列中传播，部分和留在 PE 附近累加。
- Tensor Core 利用专用矩阵阵列提高 MMA 吞吐

[查看 3×3 脉动阵列数据流动画](<../../attachments/systolic_array_3x3_dataflow.html>) （但这个动画只是通用理解模型，不代表某一代 Tensor Core 的精确内部实现）

## 数据流

一个 warp 集体参与的操作：

```
Warp Scheduler
  ↓ 发射 warp-level MMA
一个 Warp 的 32 个 Threads
  ↓ 共同提供矩阵 fragments
Tensor Core 执行 D = A × B + C
```

 >**Note** — CUDA 的 Thread/Warp/SIMT 执行模型仍然存在，只是某些矩阵指令不发给 CUDA Core，而是发给 Tensor Core。

不是“每个线程各算一个完整小矩阵”，而是整个 warp 共同完成一个矩阵 Tile。矩阵 fragment 仍然分布在 warp 的各个线程寄存器里。NVIDIA 的 WMMA 接口本身就是 warp-level MMA，fragment 的寄存器也分布在 warp 各线程中。

[Volta](<V100%20Tensor%20Core%20%E6%95%B0%E6%8D%AE%E6%B5%81.md>):
```
GMEM → REG → SMEM → REG → Tensor Core → REG
```

[Ampere](<A100%20Tensor%20Core%20%E6%95%B0%E6%8D%AE%E6%B5%81.md>)  新增了`cp.async` 路径：
```
GMEM > SMEM → REG → Tensor Core → REG
      cp.async
```

[Hopper](<H100%20Tensor%20Core%20%E6%95%B0%E6%8D%AE%E6%B5%81.md>) 常见 SS-WGMMA 路径:
```
GMEM > SMEM > Tensor Core → REG
        TMA       WGMMA
```

[Blackwell](<Blackwell%20Tensor%20Core%20%E6%95%B0%E6%8D%AE%E6%B5%81.md>)
```
GMEM > SMEM > Tensor Core > TMEM
        TMA      tcgen05.mma
                                      │
                                      └→ REG → epilogue
```
