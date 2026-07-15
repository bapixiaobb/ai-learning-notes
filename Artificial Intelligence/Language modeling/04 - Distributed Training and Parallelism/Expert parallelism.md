#AI #LanguageModeling #GPU

现在很多大模型都是 [Mixture of Experts (MoE)](<../05%20-%20Architectures%20and%20MoE/Mixture%20of%20Experts%20(MoE).md>)，这种结构能够天然的做 parallel

## Why doing EP?

Expert parallelism 和 [Tensor Parallelism](<./Tensor%20Parallelism.md>) 很像，它们都是 high bandwidth，并且能 reduce [Activations](<../../Neural%20Networks/Activations.md>) memory

但是 在 MoE layer 里，如果你既可以增加 EP degree，也可以增加 TP degree，通常先增加 EP，尽量不要过早地把单个 expert 用 TP 切得很碎。

有几个原因：
- [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 喜欢大的 matrix multiplication。[Tensor Parallelism](<./Tensor%20Parallelism.md>) degree 越高，每张 GPU 得到的矩阵越小，切得太细以后：
	- Tensor Cores 吃不满；
	- kernel launch 等固定开销占比增大；
	- GPU utilization 下降。
- MoE 的 token routing 本来就是 sparse 的
	- route sparse token activations 比传输 dense tensor-parallel matmul activations 更自然。
- MoE 已经提供了一条天然的并行轴。
	- 最自然的做法就是先把这些 experts 分给不同 GPUs，而不是先把每个 expert 自己切碎。

---
## Complexity

>**Question** — EP 局部执行难在 All-to-All；全局布局难在 DP/EP group 相互限制；模型内部又因为 EP 只处理 MLP，产生 Attention 想要 high TP、expert MLP 想要 low TP 的冲突。
#### All-to-All dispatch

MoE 中 token dispatch/combine 通常需要 All-to-All；所以这里对 communication latency 特别敏感，因为 expert 必须等待 token 到达才能计算：

```
token 还在网络上
       ↓
expert GPU 没有输入
       ↓
只能等待
```

要把这个 All-to-All 做快，需要专门的底层通信库，甚至会优化到 PTX 和硬件指令层面。比如 DeepSeek 的 **DeepEP** 和 NVIDIA 的 Hybrid EP 这类专门通信库。实现也非常复杂


#### Load imbalance

不同 expert 收到的 token 数可能不一样；这个在 [MoE imbalance mitigation](<../05%20-%20Architectures%20and%20MoE/MoE%20imbalance%20mitigation.md>) 里有细节

#### DP 和 EP 的 group 被绑在一起

旧一些的 naïve implementation 往往会用同一组 GPUs 同时承担 DP 和 EP。

例如 8 张 GPU：

```
DP：
batch 被分到这 8 张 GPU

EP：
experts 也被分到同样这 8 张 GPU
```

于是这两件事不再独立：

```
我想把 data 分成几份
        ↕ 互相限制
我想把 experts 分到几张卡
```

如果 EP group 必须嵌套在这个 DP group 里，那么：

- EP degree 受 DP group size 限制；
- 改 DP 布局会影响 token routing；
- 改 EP degree 又会影响每张卡得到的数据和 experts；
- 不能再把 DP、EP 当作完全独立的 LEGO blocks。

#### EP 只切 MLP，不切 Attention

一个 [Transformer Block](<../../Transformer/Transformer%20Block.md>) 里本来就有 [Attention](<../../Transformer/Query%20Key%20Value.md>) 和 [SwiGLU](<../../Transformer/SwiGLU.md>) MLP；MoE 只把其中“一个 SwiGLU MLP”替换成“router + 多个 SwiGLU expert”，EP 再把这些 expert 分配到不同 GPU，Attention 的结构完全没有变。

**Attention 和 MLP 的 TP 冲突：**一个 block 里虽然先做 Attention、再做 MLP，但训练系统通常要为两个部分安排并行策略：

```
同一个 Transformer block
├── Attention：可能希望使用较高 TP
└── MoE MLP：已经使用 EP，通常希望较低 TP
```

原因是：

- Attention 仍然是一个很大的 dense computation，用较高 TP 可能有帮助。
- MoE 的 expert 已经通过 EP 分到不同 GPU。
- 如果再给每个 expert 使用很高的 TP，就会把每个 SwiGLU expert 的矩阵切得太小。
- 小矩阵乘法不容易充分利用 GPU。

但是 Megatron Core 提出了一个方案： **MoE Parallel Folding**
>Attention 使用一套 parallel configuration，MoE MLP 使用另一套 parallel configuration。

它把同一批 GPU 看成两套不同的 logical groups：
- Attention：[Tensor Parallelism](<./Tensor%20Parallelism.md>) × Context Parallelism × [Data parallelism](<./Data%20parallelism.md>) × [Pipeline Parallelism](<./Pipeline%20Parallelism.md>)
- MoE MLP： Expert Tensor Parallel × Expert Parallel × Expert Data Parallel × [Pipeline Parallelism](<./Pipeline%20Parallelism.md>)
