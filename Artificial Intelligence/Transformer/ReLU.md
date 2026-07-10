#DeepLearning #NeuralNetwork #ActivationFunction

**定义**：ReLU，全称 **Rectified Linear Unit**，是一种常用的 activation function。
$$

  

\operatorname{ReLU}(x) = \max(0, x)

  

$$
也就是：
$$

  

\operatorname{ReLU}(x)

  

=

  

\begin{cases}

  

x, & x > 0 \\

  

0, & x \leq 0

  

\end{cases}

  

$$
## 直觉
ReLU 会把负数截断为 0，正数保持不变。
```text
-3 -> 0
-1 -> 0
 0 -> 0
 2 -> 2
 5 -> 5
```

## **在 Neural Network 中的作用**

如果 neural network 只有 linear layers，那么多层线性变换仍然等价于一个线性变换：

$$  
W_3W_2W_1x = Wx  
$$

因此模型表达能力仍然有限。

ReLU 的作用是在线性变换之间加入 nonlinearity：

$$  
x \mapsto \operatorname{ReLU}(Wx + b)  
$$

使 neural network 能拟合更复杂的 nonlinear function。

## **缺点**

如果某个 neuron 的输入长期小于 0，它的输出和梯度都会是 0，可能不再更新。这种现象叫做 **dead ReLU**。

## **Related**

- [[Neural Network]]
- [[Deep Learning]]
- [[Activation Function]] 
- [[Forward Propagation]]
- [[Backward Propagation]] 