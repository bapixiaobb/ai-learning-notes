#DeepLearning #NeuralNetwork #ComputationalGraph

>**Note**
> **Forward Propagation** 是指在 neural network 或 computational graph 中，从输入开始，按照模型结构一层一层向前计算，得到预测结果和 loss 的过程。

## 基本形式

给定输入 $x$ 和模型参数 $\theta$，模型先计算预测值：

```math
\hat{y} = f_\theta(x)
```

再用预测值和真实目标 $y$ 计算 loss：

```math
\mathcal{L} = \ell(\hat{y}, y)
```

## 例子：两层 Neural Network

```math
z_1 = W_1x + b_1
```
```math
a_1 = \sigma(z_1)
```
```math
z_2 = W_2a_1 + b_2
```
```math
\hat{y} = z_2
```
```math
\mathcal{L} = \ell(\hat{y}, y)
```

其中：

- $W_1, W_2$ 是 weight matrices
- $b_1, b_2$ 是 biases
- $\sigma$ 是 [Activation Function](<./Activation%20Function.md>)
- $\mathcal{L}$ 是 loss


## 直觉

  

>**Note**
> Forward propagation 就是用当前参数做一次预测，并计算预测结果有多差。
>
> input  
> → model  
> → prediction  
> → loss


>**Question**
> 在当前参数 $\theta$ 下，模型预测得怎么样？

## 在 Language Model 中

  

>**Note**
> 在 [Language Modeling](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 中，forward propagation 通常可以理解为：
>
> tokens  
> → [Embedding](<../Transformer/Embedding.md>)  
> → [Transformer](<../Transformer/Transformer.md>)  
> → logits  
> → [Softmax](<../Transformer/Softmax.md>)  
> → next-token probability  
> → [Cross Entropy Loss](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)
## **Related**

- [Neural Network](<./Neural%20Network.md>)
- Backward Propagation
- [Activation Function](<./Activation%20Function.md>)
- Loss Function
- [Transformer](<../Transformer/Transformer.md>)