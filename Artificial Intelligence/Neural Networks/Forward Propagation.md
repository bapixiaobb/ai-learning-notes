#DeepLearning #NeuralNetwork #ComputationalGraph

Forward Propagation 是给定 input 和当前 parameters，沿 [computation graph](<Computational%20Graph.md>) 正向计算 model output 的过程。

**Forward pass** 和 **forward propagation** 通常指同一个过程；它们不等于 [Feed-Forward Network](<Feed-Forward%20Network.md>) 这种 architecture。

数学上：

```math
\hat y=f_\theta(x)
```

训练时还会继续计算：

```math
\mathcal L=\ell(\hat y,y)
```

Forward 不更新 parameters。[Backpropagation](<Backpropagation.md>) 负责计算 gradients，[Optimizer](<../Transformer/Optimizer.md>) 才使用 gradients 更新 parameters。

## 基础 Neural Network

```math
z_1=W_1x+b_1
```

```math
a_1=\sigma(z_1)
```

```math
\hat y=W_2a_1+b_2
```

这里 $\sigma$ 是 [Activation Function](<Activation%20Function.md>)；$z_1$ 和 $a_1$ 是 intermediate [Activations](<Activations.md>)，$\hat y$ 是 model output。

## Forward 产生什么

一次 model forward 会得到：

1. model output，例如 prediction 或 [logits](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Logits.md>)；
2. computation graph 中的 intermediate [Activations](<Activations.md>)。

Training 时，[loss function](<Loss%20Function.md>) 会继续读取 model output 和 targets，计算 loss。广义上，人们有时把从 input 一直到 loss 的这一整段都称为 training forward phase。

    input
    → operations defined by parameters
    → intermediate activations
    → model output

    (model output, targets)
    → loss

## Transformer 只是另一张 Computation Graph

Transformer 使用相同的 forward propagation，只是 operations 变成了 embedding、attention、normalization、MLP 和 residual connections。

整体 shape flow 是：

```math
\text{token IDs }[B,S]
\rightarrow
\text{hidden states }[B,S,d_{\text{model}}]
\rightarrow
\text{logits }[B,S,V]
```

Training 时再计算：

```math
(\text{logits},\text{targets})
\rightarrow
\mathcal L
```

具体 graph 已经由 [Transformer](<../Transformer/Transformer.md>)、[Transformer Block](<../Transformer/Transformer%20Block.md>) 和 TransformerLM 图描述，这里不重复展开。

## 为什么 Forward 会占 Memory

Training 之后还要 backward，所以 forward 中一部分 [Activations](<Activations.md>) 必须暂时保留。

[Recomputation](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Recomputation.md>) 会少保存一些 activations，在 backward 时重新执行部分 forward，用 compute 换 memory。

在 [Data parallelism](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/Data%20parallelism.md>) 中，每个 rank 都为自己的 local batch 执行 forward；[ZeRO](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/ZeRO.md>) 可以 shard model states，但不会自动切分 local activations。

>**Summary**
> Forward propagation 是沿 computation graph 计算 $f_\theta(x)$，得到 model output 和 intermediate activations。
> Training 时，loss function 再把 output 和 targets 接到这张 graph 后面。

## Related

- [Activations](<Activations.md>)
- [Activation Function](<Activation%20Function.md>)
- [Computational Graph](<Computational%20Graph.md>)
- [Backpropagation](<Backpropagation.md>)
- [Feed-Forward Network](<Feed-Forward%20Network.md>)
- [Transformer](<../Transformer/Transformer.md>)
- [Transformer Block](<../Transformer/Transformer%20Block.md>)
- [Recomputation](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Recomputation.md>)
