#DeepLearning #NeuralNetwork

**定义**：Activation Function 是 [[Neural Network]] 中作用在每一层输出上的非线性函数，用来增强模型的表达能力。

在线性层之后，通常会加入 activation function：

```math
z = Wx + b
```

```math
a = \sigma(z)
```

其中 $\sigma$ 表示 activation function。

## 直觉

如果 neural network 只有 linear layers，那么多层线性变换仍然等价于一个线性变换：

```math
W_3W_2W_1x = Wx
```

因此模型无法表示复杂的 nonlinear function。

Activation function 的作用是引入 nonlinearity，使 neural network 能拟合更复杂的函数关系。

## 常见 Activation Functions

| Activation Function | 形式                                                                         | 特点                                    |
| ------------------- | -------------------------------------------------------------------------- | ------------------------------------- |
| [[ReLU]]            | $\max(0,x)$                                                              | 简单、高效，常用于基础 neural network            |
| [[Sigmoid]]         | $\frac{1}{1+e^{-x}}$                                                     | 输出在 $(0,1)$，容易出现 vanishing gradient |
| Tanh                | $\tanh(x)$                                                               | 输出在 $(-1,1)$                        |
| GELU                | 平滑的 nonlinear activation                                                   | Transformer 中常见                       |
| [[SiLU]] / Swish    | $x \cdot \operatorname{sigmoid}(x)$                                      | 现代深度模型中常见                             |
| [[GLU]]             |                                                                            | gated linear uint                     |
| [[SwiGLU]]          | $\mathrm{SwiGLU}(x) = W_2 \left( \mathrm{SiLU}(W_1 x) \odot W_3 x \right)$ | Transformer FFN 里的 gated activation   |

## Related

- [[Neural Network]]
- [[Deep Learning]]
- [[ReLU]]
- [[Forward Propagation]]
- [[Backward Propagation]]