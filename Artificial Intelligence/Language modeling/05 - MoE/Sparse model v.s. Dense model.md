#AI #LanguageModeling


- **Dense model:** In dense models all the parameters are used for all the inputs.
	- Dense FFN：不同 tokens 都经过同一个 FFN，使用同一组参数。
	- 一般我们讨论的 dense model 可以参考 [Llama-style Architecture](<../../Transformer/00%20-%20Maps%20and%20Architectures/Llama-style%20Architecture.md>)
- **Sparse model:** Sparsity uses the idea of **conditional computation**
	- Sparse [MoE](<Mixture%20of%20Experts.md>)：有多组 expert 参数，但每个 token 只选择其中几组参与计算
```math
\text{more total parameters}
\quad
\text{without proportional increase in active computation}
```

## 🌱 Sparsity ([What is sparsity?](https://huggingface.co/blog/moe#what-is-sparsity))
MoE 的 sparsity 指：每个 token 只激活全部 experts 中的一小部分。
例如，一个 MoE layer 有 64 个 experts，但每个 token 只选择 2 个 experts：
```math
64 \text{ experts total}
```
```math
2 \text{ experts active per token}
```
这是模型有很多 total parameters，但每个 token 的 active parameters 较少。
>**Note**
>Sparsity 让模型可以扩大参数容量，同时控制每个 token 的计算量。

这也是 MoE 和 [dense Transformer](<../../Transformer/00%20-%20Maps%20and%20Architectures/Llama-style%20Architecture.md>) 的关键区别：

| **Model Type**   | **每个 token 使用的参数**    |
| ---------------- | --------------------- |
| dense model      | 通常使用整层参数              |
| sparse MoE model | 只使用被 route 到的 experts |

---
## 🧪 MoE vs Dense FFN

| **Aspect**               | **Dense FFN**        | **MoE Layer**                      |
| ------------------------ | -------------------- | ---------------------------------- |
| parameters               | 所有 token 共享同一个 FFN   | 多个 experts                         |
| computation              | 每个 token 用同一组 FFN 参数 | 每个 token 只用 top-k experts          |
| capacity                 | 参数量和 compute 通常一起增加  | total parameters 可以大幅增加            |
| [routing](<MoE%20Router.md>) | no routing           | router / gate 决定 expert assignment |
| challenge                | 相对简单稳定               | routing imbalance、通信、训练复杂度         |
