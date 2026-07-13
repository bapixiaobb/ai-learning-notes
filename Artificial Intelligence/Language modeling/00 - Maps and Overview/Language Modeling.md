#AI #MachineLearning #DeepLearning #NLP

Language Modeling 是一个目标，不等于 Transformer

>**Note** — Language Modeling 的核心定义是
>给定前面的 token，预测下一个 token

所以 language modeling 是一个建模目标，不是某一种固定架构

---
**Language Modeling** 是建模 token sequence 概率分布的任务。

给定一个 token 序列：

```math
x_1, x_2, \dots, x_T
```

language model 使用 chain rule 建模：

```math
p(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T}
p(x_t \mid x_{<t})
```
## Core Objective

现代 autoregressive language model 的目标是：

```math
p_\theta(x_t \mid x_1, \dots, x_{t-1})
```

也就是根据前面的 tokens 预测下一个 token。

## Architecture
- 什么是 [Language Model Architecture](<../05%20-%20Architectures%20and%20MoE/Language%20Model%20Architecture.md>)？
- 为什么现代 language model 通常使用 [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>)？
- 一个 [Transformer Block](<../../Transformer/Transformer%20Block.md>) 由哪些部分组成？
- 什么是 [Self-Attention](<../../Transformer/Self-Attention.md>)？
- 什么是 [Multi-Head Attention](<../../Transformer/Multi-Head%20Attention.md>)？
- 为什么 autoregressive model 需要 [Causal Mask](<../../Transformer/Causal%20Mask.md>)？
- token 的顺序信息如何通过 [Positional Encoding](<../../Transformer/Positional%20Encoding.md>) 表示？
- 什么是 [Rotary Position Embedding](<../../Transformer/Rotary%20Position%20Embedding.md>)？
- 为什么 Transformer block 需要 [Residual Connection](<../../Transformer/Residual%20Connection.md>)？
- 为什么深层 Transformer 需要 [Layer Normalization](<../../Transformer/Layer%20Normalization.md>) 或 [RMSNorm](<../../Transformer/RMSNorm.md>)？
- 什么是 [Pre-Norm Transformer](<../../Transformer/Pre-Norm%20Transformer.md>)？
- [MLP](<../../Transformer/MLP.md>) / [Feed-Forward Network](<../../Neural%20Networks/Feed-Forward%20Network.md>) 在 Transformer 中做什么？
- 什么是 Language Modeling Head？
## **Pipeline**

```text
raw text
-> [Tokenization](<../01%20-%20Language%20Modeling%20Basics/Tokenization.md>)
-> token IDs
-> embeddings
-> [Transformer](<../../Transformer/Transformer.md>)
-> logits
-> softmax
-> next-token probability
```
## **Related**

- [Natural Language Processing](<../../Fundamentals/Natural%20Language%20Processing.md>)
- [Deep Learning](<../../Neural%20Networks/Deep%20Learning.md>)
- [Tokenization](<../01%20-%20Language%20Modeling%20Basics/Tokenization.md>)
- [Transformer](<../../Transformer/Transformer.md>)
- [Cross Entropy Loss](<../01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)
- [Scaling Law](<../02%20-%20Training%20and%20Scaling/Scaling%20Law.md>)
- [Language Model Architecture](<../05%20-%20Architectures%20and%20MoE/Language%20Model%20Architecture.md>)