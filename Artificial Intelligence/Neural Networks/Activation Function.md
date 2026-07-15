#DeepLearning #NeuralNetwork

Activation Function 是 [Neural Network](<./Neural%20Network.md>) 中的 nonlinear operation。

>**Important**
> Activation Function 不等于 [Activations](<./Activations.md>)。
> 前者是模型执行的函数；后者是 forward 过程中产生的 intermediate tensors。

一层基础 neural network 常写成：

```math
z=Wx+b
```

```math
a=\sigma(z)
```

其中 $\sigma$ 是 activation function，$a$ 是这个 operation 产生的一个 activation。

## 为什么需要 Nonlinearity

如果模型只有 linear layers，多层线性变换仍然可以合并成一次线性变换：

```math
W_3W_2W_1x=Wx
```

Activation function 引入 nonlinearity，使模型能够表达更复杂的函数。

## 常见 Activation Functions

| Function | 形式 | 常见位置 |
|---|---|---|
| [ReLU](<../Transformer/ReLU.md>) | $\max(0,x)$ | 基础 neural network |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | gating / probability-style output |
| Tanh | $\tanh(x)$ | 早期 neural network |
| GELU | smooth nonlinear function | Transformer MLP |
| [SiLU](<../Transformer/SiLU.md>) | $x\cdot\operatorname{sigmoid}(x)$ | modern neural network |
| [SwiGLU](<../Transformer/SwiGLU.md>) | gated nonlinear transformation | modern Transformer MLP |

## 在 Transformer 中

Transformer 的 activation function 主要出现在 MLP / FFN 中，例如 GELU 或 SwiGLU。

Attention、normalization 和 residual addition 也会产生 [Activations](<./Activations.md>)，但它们本身不是 activation functions。

>**Summary**
> Activation function 回答的是“模型用了什么 nonlinear operation”；[Activations](<./Activations.md>) 回答的是“forward 产生了什么数据”。

## Related

- [Activations](<./Activations.md>)
- [Forward Propagation](<./Forward%20Propagation.md>)
- [Transformer Block](<../Transformer/Transformer%20Block.md>)
- [MLP](<../Transformer/MLP.md>)
