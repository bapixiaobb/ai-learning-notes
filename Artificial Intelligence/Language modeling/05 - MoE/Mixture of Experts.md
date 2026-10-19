#AI #Transformer

## Why we Need MoE?

增大 model 的参数量，是提升 model capacity 的一种方式。但对于 [dense model](<Sparse%20model%20v.s.%20Dense%20model.md>)，增大 [FFN](<../../Transformer/03%20-%20MLP%20and%20Activations/Multilayer%20Perceptron.md>) 通常也会增加每个 token 的 compute cost。

**MoE 提供了另一种 scale model 的方式：增加总参数量，但让每个 token 只使用其中一部分来控制 compute cost。**

>**Important** — MoE 可以在拥有更多 total parameters 的同时，让每 token 的 compute 与较小的 dense model 接近。
>所以模型的整体容量可以非常大

## What is MoE?

在 [Transformer](<../../Transformer/00%20-%20Maps%20and%20Architectures/Transformer%20Block.md>) 中，常见的做法是把部分或全部 block 中的 [dense FFN](<../../Neural%20Networks/Feed-Forward%20Network.md>) 替换成 MoE layer。

```math
\text{Attention}
\rightarrow
\text{MLP / FFN}  \Longrightarrow \text{Attention}
\rightarrow
\text{MoE Layer}
```

一个 MoE layer 主要包含两部分：
- **Experts**
	- 真正处理 token 的多个 [FFN](<../../Neural%20Networks/Feed-Forward%20Network.md>) / [MLP](<../../Transformer/03%20-%20MLP%20and%20Activations/Multilayer%20Perceptron.md>)。每个 expert 都有自己的一组参数。
	- 每个 expert 通常本身就是一个 FFN / MLP：
```math
\text{Expert}_i(x)=
\mathrm{FFN}_i(x)
```
- **[Router / gate](<MoE%20Router.md>)**
	- 根据 token 当前的 representation，决定让哪些 experts 参与计算
	- Router 本身也有可训练参数，会和 model 的其他部分一起学习。

对于每个 token，router 通常只选择少数几个 experts，再把它们的输出加权组合。这种根据输入选择部分计算路径的方式，就是 **conditional computation**。

因此，MoE 可以拥有较大的 **total parameters**，同时保持较少的 **active parameters per token**。但少算一部分参数，不代表这些参数不需要存储：MoE 在控制 compute cost 的同时，仍需要为整个 model 的参数提供存储空间。

![MoE](<../../attachments/MoE.png>)

## ⚠️ Main Challenges

MoE 的难点不在于概念，而在于 routing 和 systems

- **Routing 的训练**：top-k selection 是离散选择，router 和 experts 如何获得训练信号？见 [MoE Router](<MoE%20Router.md>)。
- **Load imbalance**：不同 experts 收到的 tokens 数量不均，可能影响训练和设备利用率。见 MoE imbalance。
- **Communication cost**：experts 分布在不同 devices 时，需要传输 token activations 和计算结果。见 [Expert parallelism](<../04%20-%20Distributed%20Training%20and%20Parallelism/Expert%20parallelism.md>)。
