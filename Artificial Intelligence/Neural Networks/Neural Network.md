#AI #MachineLearning #DeepLearning #Optimization

**定义**：Neural Network 是一种参数化函数模型，用许多可学习参数 $\theta$ 构造函数 $f_\theta$，通过训练数据调整参数，使模型输出尽量接近目标。
```math
\hat{y} = f_\theta(x)
```

从 Statistical Learning 的角度看，neural network 是一个可学习的 hypothesis class；从 An Optimization Problem 的角度看，训练 neural network 是在最小化 loss function：

```math
\min_\theta \mathcal{L}(\theta)
```
其中：
- $\theta$：模型中的所有 weights 和 biases
- $f_\theta$：neural network 表示的函数
- $\mathcal{L}(\theta)$：衡量预测误差的 loss function

## 与 Regression 的关系

Neural network 可以看作 nonlinear regression 的一种形式。

在 Regression Analysis 中，linear regression 通常写成：

```math
\hat{y} = Wx + b
```

它只能表示 linear mapping。

Neural network 在 linear transformations 之间加入 nonlinear activation functions：

```math
f_\theta(x)
=
W_L \sigma(
W_{L-1} \sigma(
\cdots
\sigma(W_1x + b_1)
)
+ b_{L-1}
)
+ b_L
```

其中 $\sigma$ 是 activation function，例如 [ReLU](<../Transformer/ReLU.md>)。

如果没有 activation function，多层 linear layers 仍然只是一个 linear transformation：

```math
W_3W_2W_1x = Wx
```

因此 neural network 的表达能力主要来自：

```text
linear transformation + nonlinear activation
```

## **训练过程**

Neural network 的训练过程通常包括：

1. 使用 [Forward Propagation](<./Forward%20Propagation.md>) 计算 prediction；
2. 使用 loss function 比较 prediction 和 target，得到 loss；
3. 使用 [Backpropagation](<./Backpropagation.md>) 计算 loss 对参数的 gradients；
4. 使用 [Optimizer](<../Transformer/Optimizer.md>) 更新参数。

整体过程可以写成：
```math
\hat{y} = f_\theta(x)
```
```math
\mathcal{L} = \ell(\hat{y}, y)
```
```math
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)
```
其中 $\eta$ 是 learning rate。

## **为什么 Neural Network 能拟合复杂函数**

Neural network 是由多个 affine transformations 和 nonlinear activations 复合而成的函数。

## $f_\theta$

```math
f_L \circ f_{L-1} \circ \cdots \circ f_1
```

这种复合结构使它能够表示复杂 nonlinear mappings。

对于 ReLU neural network，可以直观理解为：

```text
linear regression: 一条直线或一个线性超平面
ReLU neural network: 很多 piecewise linear regions 拼接起来的函数
```

## **与 Linear System 的区别**

Linear System 通常研究的是：

```math
Ax = b
```

其中矩阵 $A$ 是给定的，目标是求未知向量 $x$。

Neural network training 中，模型参数 $\theta$ 是未知的，并且 loss function 通常是 nonlinear and nonconvex：
```math
\min_\theta \mathcal{L}(\theta)
```
因此 neural network training 更接近 large-scale nonlinear optimization，而不是直接求解一个固定 linear system。

## **任务类型**

Neural network 可以用于不同类型的任务：

|**任务**|**输出**|**常用 Loss**|
|---|---|---|
|Regression|连续值|Mean Squared Error|
|Classification|类别概率|Cross Entropy Loss|
|Language Modeling|next-token probability|Cross Entropy Loss|
|Function Approximation|函数值|MSE 或 residual loss|

在 [Language Modeling](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 中，neural network 通常用于建模：

```math
p_\theta(x_t \mid x_{<t})
```
即根据前面的 tokens 预测下一个 token。

## **Related**

- [Deep Learning](<./Deep%20Learning.md>)
- Regression Analysis
- Statistical Learning
- An Optimization Problem
- [Forward Propagation](<./Forward%20Propagation.md>)
- [Backpropagation](<./Backpropagation.md>)
- Gradient Descent
- [ReLU](<../Transformer/ReLU.md>)
- [Transformer](<../Transformer/Transformer.md>)
