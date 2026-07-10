#DeepLearning #NeuralNetwork #ComputationalGraph

>[!note]
> **Forward Propagation** 是指在 neural network 或 computational graph 中，从输入开始，按照模型结构一层一层向前计算，得到预测结果和 loss 的过程。

## 基本形式

给定输入 $x$ 和模型参数 $\theta$，模型先计算预测值：

$$
\hat{y} = f_\theta(x)
$$

再用预测值和真实目标 $y$ 计算 loss：

$$
\mathcal{L} = \ell(\hat{y}, y)
$$

## 例子：两层 Neural Network

$$
z_1 = W_1x + b_1
$$
$$
a_1 = \sigma(z_1)
$$
$$
z_2 = W_2a_1 + b_2
$$
$$
\hat{y} = z_2
$$
$$
\mathcal{L} = \ell(\hat{y}, y)
$$

其中：

- $W_1, W_2$ 是 weight matrices
- $b_1, b_2$ 是 biases
- $\sigma$ 是 [[Activation Function]]
- $\mathcal{L}$ 是 loss


## 直觉

  

>[!note]
> Forward propagation 就是用当前参数做一次预测，并计算预测结果有多差。
> 
> input  
> → model  
> → prediction  
> → loss


>[!question]  
> 在当前参数 $\theta$ 下，模型预测得怎么样？

## 在 Language Model 中

  

>[!note]
> 在 [[Language Modeling]] 中，forward propagation 通常可以理解为：
> 
> tokens  
> → [[Embedding]]  
> → [[Transformer]]  
> → logits  
> → [[Softmax]]  
> → next-token probability  
> → [[Cross Entropy Loss]]
## **Related**

- [[Neural Network]]
- [[Backward Propagation]]
- [[Activation Function]]
- [[Loss Function]]
- [[Transformer]] 