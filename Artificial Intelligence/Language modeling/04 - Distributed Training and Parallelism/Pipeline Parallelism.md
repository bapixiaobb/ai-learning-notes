#AI #LanguageModeling #GPU

# Layer-wise parallel

![layer-wise parallel.png](<../../attachments/layer-wise%20parallel.png>)

Pipeline parallelism 把连续的 layer groups 分成 stages，放到不同的 GPUs 上。

- [Activations](<../../Neural%20Networks/Activations.md>) 在 [forward pass](<../../Neural%20Networks/Forward%20Propagation.md>) 中从前一 stage 传给后一 stage；
- activation gradients $\partial \mathcal L/\partial h$ 在 [backward pass](<../../Neural%20Networks/Backpropagation.md>) 中从后一 stage 传回前一 stage。

每个 stage 再根据自己的 inputs 和收到的 activation gradients，计算本地 parameter gradients。

# Problem

想象一下，如果做 layer-wise parallel 的话，GPUs 的并行度是很糟糕的，因为如下图所示
![layer-wise parallel 1 batch.png](<../../attachments/layer-wise%20parallel%201%20batch.png>)
F 是 [Forward Propagation](<../../Neural%20Networks/Forward%20Propagation.md>)，B 是 [Backpropagation](<../../Neural%20Networks/Backpropagation.md>)，这几个不同颜色的块就代表了不同的 GPUs，一行代表了一张卡
在一层一层 layers 传递的过程中，需要等上一层算完，再传递给下一层，这样是串行的，和我们说的 [GPU 核心思想](<../03%20-%20GPU%20and%20Systems/GPU%20%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3.md>) 不一样

## Solution

有一个办法就是用 micro-batches
![layer-wise parallel micro-batches.png](<../../attachments/layer-wise%20parallel%20micro-batches.png>)
如上图所示，把一个 batch 切成小的 micro-batches，在第一个 GPU 算完第一个 micro-batch 的时候，立马算 第二个 micro-batch，然后同时第二个 GPU 算第一个 micro-batch

上面空白的，没有任何计算的时候，就是 **Bubble**

我们假设：
- `p`：pipeline stages 数量，也就是把模型切成几段
- `m`：micro-batches 数量
- `τ`：一个 stage 处理一个 micro-batch 的时间
👆这张图，单论 Forward，每一张 GPU（也就是每一行）
- 真正计算了 `m=4` 格：useful compute = `4τ`
- 因为 pipeline 的填充和排空，空了 `p-1=3` 格：bubble time = `3τ`
所以：
```math
\frac{\text{bubble time}}{\text{useful compute}} = \frac{(p-1)\tau}{m\tau} = \frac{p-1}{m}
```
所以我们需要切多一些 batches 来藏 bubble，当然 micro-batch 不能无限切小，否则单个 micro-batch 太小，GPU 本身吃不满，通信和 kernel launch 的固定开销也会变得明显

---
# Why pipeline parallel

如果 pipeline parallel 有这么大的问题，为什么还用它？

- 相比于 [Data parallelism](<./Data%20parallelism.md>)，显存需求低：切了 [Activations](<../../Neural%20Networks/Activations.md>)
- communication properties good: 只在相邻 stage 之间做 point-to-point communication，不需要每次都让整个 group 一起通信。
```
GPU 0 ──activations──> GPU 1 ──activations──> GPU 2
GPU 0 <──activation gradients── GPU 1 <──activation gradients── GPU 2
```
>**Note** — 因此，当机器跨越较慢的连接，例如跨 node、跨 rack 时，pipeline 可能更适合放在这些慢链路上。

**Pipeline parallelism 不是为了让单张卡算得更快。它接受 bubble 带来的计算利用率损失，换取“更大的模型能运行”，并让跨慢速网络的通信更加可控。**

---
# 一种调度技巧：[Zero Bubble Pipelining](<./Zero%20Bubble%20Pipelining.md>)