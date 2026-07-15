#DeepLearning #NeuralNetwork #LanguageModeling #Systems

Activations 是模型处理当前 input 时，在 [Forward Propagation](<./Forward%20Propagation.md>) 中产生的 intermediate tensors。

它们不一定经过 [Activation Function](<./Activation%20Function.md>)。

## 从基础 Neural Network 理解

```math
z_1=W_1x+b_1
```

```math
a_1=\sigma(z_1)
```

```math
z_2=W_2a_1+b_2
```

基础教材通常把 $a_1$ 称为 activation，因为它是 activation function 的输出。

但从 [computation graph](<./Computational%20Graph.md>) / systems 的角度看，$z_1$、$a_1$ 和 $z_2$ 都是 forward 中产生的 intermediate activations。

## Transformer 中的 Activations

Transformer 处理的是一系列 tensor representations。常见 activations 包括：

| Tensor | 典型 shape | 含义 |
|---|---|---|
| block hidden states | $[B,S,d_{\text{model}}]$ | 每个 token 当前的 representation |
| $Q,K,V$ | $[B,h,S,d_{\text{head}}]$ | attention 的中间 tensors |
| attention output | $[B,S,d_{\text{model}}]$ | attention 对 residual stream 的更新 |
| MLP intermediate | $[B,S,d_{\text{ff}}]$ | MLP 扩展维度后的中间 tensor |

这些数据来自 [Transformer Block](<../Transformer/Transformer%20Block.md>) 中的 embedding、attention、normalization、MLP 和 residual operations。只有其中某些 operations 使用 activation function，但它们产生的 intermediate tensors 都可以称为 activations。

## Parameters 和 Activations

| | Parameters | Activations |
|---|---|---|
| 来源 | 训练得到的 model weights | 当前 batch 经过 forward 产生 |
| 是否随 input 改变 | 不随当前 input 改变 | 每个 batch 都不同 |
| 例子 | $W_Q,W_K,W_V,W_1,W_2$ | hidden states、$Q/K/V$、MLP intermediate |
| 用途 | 定义模型怎样计算 | 承载当前 input 的数据流 |

## 为什么 Training 要保存 Activations

[Backpropagation](<./Backpropagation.md>) 沿 computation graph 反向计算 gradients 时，需要 forward 中的一部分 inputs 或 intermediate results。

因此 training forward 不能立即释放所有 activations。并不是每个临时 tensor都会被保存；只需要保留 backward 所需的部分。

[Recomputation](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Recomputation.md>) / activation checkpointing 会少保存一些 activations，在 backward 时重新计算：

```math
\text{activation memory}\downarrow,
\qquad
\text{compute}\uparrow
```

## Related

- [Activation Function](<./Activation%20Function.md>)
- [Forward Propagation](<./Forward%20Propagation.md>)
- [Backpropagation](<./Backpropagation.md>)
- [Transformer Block](<../Transformer/Transformer%20Block.md>)
- [Recomputation](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Recomputation.md>)
- [Resource Accounting](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)
- [ZeRO](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/ZeRO.md>)
