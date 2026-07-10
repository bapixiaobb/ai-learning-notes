
#DeepLearning #LanguageModeling #Systems #Optimization

>[!note]
> **Scaling Law** 描述的是：当模型规模、数据规模和计算量增加时，模型性能如何系统性变化。
>
> 在 language model 中，scaling law 常用于预测：给定训练 compute budget，应该选择多大的模型、多少训练 tokens，以及最终 loss 大概会是多少。

## 核心问题

>[!question]
> 如果我们增加 model parameters、training tokens 或 training compute，language model 的 loss 会如何变化？

常见变量包括：

| 符号 | 含义 |
|---|---|
| $N$ | number of model parameters |
| $D$ | number of training tokens |
| $C$ | training compute |
| $L$ | loss |

## 基本直觉

>[!note]
> 一般来说，模型越大、训练 tokens 越多、训练 compute 越高，loss 通常会下降。
>
> 但这种下降不是线性的，而是近似 obey power-law behavior。

可以粗略理解为：

```math
L(C)
\approx
A C^{-\alpha} + L_\infty
```

其中：

- $C$：training compute
- $L(C)$：给定 compute 下的 loss
- $A, \alpha$：拟合出来的常数
- $L_\infty$：不可约误差或极限 loss

## 与 Training Compute 的关系

>[!note]
> 在 dense Transformer training 中，training compute 常用近似：
>
> ```math
> C \approx 6ND
> ```
>
> 其中 $N$ 是参数量，$D$ 是训练 token 数。

因此 scaling law 关心的不只是总 compute $C$，还关心：

>[!question]
> 给定同样的 compute budget，应该把 compute 用在更大的模型上，还是更多的 training tokens 上？

## Compute-optimal Training

>[!note]
> **Compute-optimal training** 研究的是：在固定 compute budget 下，如何选择模型大小 $N$ 和训练 token 数 $D$，使最终 loss 最低。

也就是求：

```math
\min_{N,D} L(N,D)
```

subject to：

```math
C \approx 6ND
```

>[!note]
> 如果模型太大但 tokens 太少，模型可能 under-trained；如果 tokens 太多但模型太小，模型 capacity 可能不足。Compute-optimal training 试图在二者之间找到平衡。

## IsoFLOP Analysis

>[!note]
> **IsoFLOP analysis** 是在固定 compute budget 下，比较不同 model size 和 token count 的训练结果。
>
> 它用于估计在同样 FLOPs 下，哪种配置最优。

例如，在同样 compute 下可以比较：

| 配置 | Model size | Training tokens |
|---|---:|---:|
| A | small model | many tokens |
| B | medium model | medium tokens |
| C | large model | fewer tokens |

>[!note]
> 如果某个配置在相同 FLOPs 下得到更低 loss，它就更接近 compute-optimal。

## 为什么 Scaling Law 重要

>[!note]
> 大模型训练成本极高，不能直接盲目训练。Scaling law 允许研究者先跑小规模实验，再预测大规模训练结果。

它可以用于：

- 预测大模型最终 loss
- 选择 model size
- 选择 training token count
- 估算 training compute
- 比较不同训练配置
- 判断训练计划是否值得投入资源

## 与 Resource Accounting 的关系

>[!note]
> [[Resource Accounting]] 关注训练需要多少 compute、memory、bandwidth 和时间；Scaling law 关注这些 compute 投入后能换来多少 loss improvement。
>
> 前者回答“训练要花多少资源”，后者回答“这些资源是否值得”。

二者结合起来就是：

```text
compute budget
-> choose model size and data size
-> estimate training cost
-> forecast final loss
```

## **与 Language Modeling 的关系**

>[!note]  
> Language model 的 loss 通常是 [[Cross Entropy Loss]] 或 negative log-likelihood。
> 
> Scaling law 研究的是当模型和数据规模扩大时，这个 loss 如何下降。

在 autoregressive language modeling 中，训练目标是：
```math
-\sum_{t=1}^{T}  
\log p_\theta(x_t \mid x_{<t})  
```

Scaling law 用来预测随着 $N$、$D$、$C$ 增加，这个 loss 会如何变化。

## **直觉总结**

> [!note]  
> Scaling law 可以理解为大模型训练中的经验规律：
> 
> 更多参数、更大数据、更多计算通常会带来更低 loss，但收益递减。
> 
> 它的价值在于帮助我们在训练前做预测，而不是等昂贵的大训练结束后才知道结果。


## **Related**

- [[Language Modeling]]
- [[Large Language Model (LLM)]]
- [[Resource Accounting]]
- [[Training Compute - 6ND]]
- [[FLOPs]]
- [[Cross Entropy Loss]] 