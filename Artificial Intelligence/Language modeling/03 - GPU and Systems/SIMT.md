#AI #GPU

>**Question** — What is SIMT
>**Single Instruction, Multiple Threads**

## Core

SIMT 的核心是：一个 warp 里的多个 thread，一起执行同一条 instruction，但处理不同的数据

[GPU](<./GPU.md>) 上这么多 thread，到底是怎么一起执行的？

>**Example**
>```
>thread 0 算 x[0]
>thread 1 算 x[1]
>thread 2 算 x[2]
>```
>它们执行的是同一条逻辑：$y[i] = x[i] * 2$，只是每个 thread 的 `i` 不一样。

kernel 要尽量写成“很多 thread 做同一种事”，GPU 才舒服。

---
# SIMT 对优化意味着什么？
## Control divergence

>**Example**
>如果一个 warp 里的 thread 走不同分支：
>```
>thread 0~15 走 if
>thread 16~31 走 else
>```
>GPU 不能真的让它们完全独立执行。

这样效率就掉了

> 所以 SIMT 下优化点之一是：**让同一个 warp 里的 thread 尽量执行相同路径，避免 warp divergence**


下面这张图可以说明，如果走分支的话，可能会出现空块，会导致串行，我们尽量要让它并行
![GPU control divergence.png](<../../attachments/GPU%20control%20divergence.png>)
所以 [GPU](<./GPU.md>) 会有比较特殊的编码形式就是 mask，用 mask 做 matrix multiplication，比选择 conditional 的编码形式要对 GPU 更友好。

## Use more Warp

因为 GPU 靠切换 warp 来 hide latency。

所以如果可运行的 warp 太少，会出现：Warp A 等 memory，Warp B 也等 memory，没有 Warp C 可切，SM 就空转了。