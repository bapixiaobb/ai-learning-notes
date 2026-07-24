#DeepLearning #LanguageModeling #Systems

从数学上看，训练一个 language model 就是在求解：

```math
\min_\theta \mathcal L(\theta)
```

一次 training step 大致是：

batch
→ [forward](<../../Neural%20Networks/Forward%20Propagation.md>)
→ loss
→ [backward](<../../Neural%20Networks/Backpropagation.md>) 得到 gradients
→ [optimizer](<../../Transformer/Optimizer.md>) 更新 parameters

如果完全不考虑机器，到这里就结束了。

但真正训练时，这些计算必须落到硬件上：parameters、[activations](<../../Neural%20Networks/Activations.md>)、gradients 和 optimizer states 要放进 memory；matrix multiplications 要交给 compute units；不同 devices 上的局部结果还要互相传输。

>**Note**
> **Systems for Language Models 研究的不是另一个模型，而是同一个数学模型怎样在现实硬件上被执行。**

# 从 Model 到 Hardware

[Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 定义目标
            ↓
[Transformer](<../../Transformer/Transformer.md>) architecture 定义 model forward 的 [computation graph](<../../Neural%20Networks/Computational%20Graph.md>)
            ↓
[Training Recipe](<../02%20-%20Training%20and%20Scaling/Training%20Recipe.md>) 定义怎样更新 parameters
            ↓
[Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 数清 compute / memory / data movement
            ↓
[GPU](<GPU.md>) / cluster 提供实际资源
            ↓
[Parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Parallelism.md>) 把 computation 和 model states 分配到多个 devices

这几层不是互相独立的：

- model architecture 决定有哪些 operators、tensor shapes 和 layer dependencies；
- training recipe 决定 [batch size](<../01%20-%20Language%20Modeling%20Basics/Batch%20Size.md>)、optimizer states 和 precision；
- 这些选择共同决定需要多少 compute、memory 和 communication；
- hardware 和 topology 决定哪种执行方式更高效。

# 核心问题

从 systems 的角度看，一次训练主要问三个问题：

1. **算得完吗？** compute 是否足够，昂贵的 compute units 有没有一直工作。
2. **放得下吗？** parameters、gradients、optimizer states 和 activations 是否能放进 memory。
3. **数据送得到吗？** memory bandwidth 和 device 间 communication 能否及时把数据送到需要它的地方。

>**Note**
> 单卡优化主要处理 compute ↔ memory 之间的数据移动；多卡训练进一步处理 GPU ↔ GPU、node ↔ node 之间的数据移动。

# 为什么会出现 Parallelism

当一张 GPU 的 memory 或 compute 不够时，我们并不是重新定义一个模型，而是把原来的训练计算拆开：

- batch 中样本梯度的求和可以拆成 [Data parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Data%20parallelism.md>)；
- matrix multiplication / hidden dimension 可以拆成 tensor parallelism；
- 多层函数的复合可以拆成 pipeline parallelism；
- parameters、gradients 和 optimizer states 可以用 ZeRO / FSDP 分片。

切开以后，每个 rank 只有局部信息，因此需要 collective communication 恢复下一步计算所需要的信息。

所以：

>**Summary**
> **Parallelism 是 execution plan；collective communication 是拆分之后重新连接信息的方式；底层 topology 决定这种连接有多贵。**

---
# 🔗

[Language Model Architecture](<../05%20-%20Architectures%20and%20MoE/Language%20Model%20Architecture.md>)
[Training Recipe](<../02%20-%20Training%20and%20Scaling/Training%20Recipe.md>)
[Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)
[GPU](<GPU.md>)
[Parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Parallelism.md>)
[GPU Communication Topology](<../04%20-%20Distributed%20Training%20and%20Parallelism/GPU%20Communication%20Topology.md>)
[Model FLOPs Utilization](<Model%20FLOPs%20Utilization.md>)
