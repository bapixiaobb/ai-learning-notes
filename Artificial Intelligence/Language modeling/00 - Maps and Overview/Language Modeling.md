#AI #MachineLearning #DeepLearning #NLP

Language Modeling 是一个目标，不等于 Transformer

>[!note] Language Modeling 的核心定义是
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
- 什么是 [[Language Model Architecture]]？
- 为什么现代 language model 通常使用 [[Decoder-Only Transformer]]？
- 一个 [[Transformer Block]] 由哪些部分组成？
- 什么是 [[Self-Attention]]？
- 什么是 [[Artificial Intelligence/Transformer/Multi-Head Attention]]？
- 为什么 autoregressive model 需要 [[Causal Mask]]？
- token 的顺序信息如何通过 [[Positional Encoding]] 表示？
- 什么是 [[Rotary Position Embedding]]？
- 为什么 Transformer block 需要 [[Residual Connection]]？
- 为什么深层 Transformer 需要 [[Layer Normalization]] 或 [[RMSNorm]]？
- 什么是 [[Pre-Norm Transformer]]？
- [[MLP]] / [[Feed-Forward Network]] 在 Transformer 中做什么？
- 什么是 [[Language Modeling Head]]？
## **Pipeline**

```text
raw text
-> [[Tokenization]]
-> token IDs
-> embeddings
-> [[Transformer]]
-> logits
-> softmax
-> next-token probability
```
## **Related**

- [[Natural Language Processing]] 
- [[Deep Learning]]
- [[Tokenization]]
- [[Transformer]]
- [[Cross Entropy Loss]]
- [[Scaling Law]] 
- [[Language Model Architecture]] 