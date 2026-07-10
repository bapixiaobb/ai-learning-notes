#DeepLearning #NeuralNetwork #MachineLearning

Deep Neural Network，简称 DNN，指包含多层 hidden layers 的 [[Neural Network]]。  
它通过多层 nonlinear transformations，把原始输入逐步变换成更抽象、更适合任务的 representation。

## 🧠 Core Idea

>[!note]
>Deep neural network 的核心思想是：
>
>用很多层函数复合来学习复杂模式。
>
>每一层都对上一层的 representation 做变换。

可以写成：

```math
x
\rightarrow
h^{(1)}
\rightarrow
h^{(2)}
\rightarrow
\cdots
\rightarrow
h^{(L)}
\rightarrow
y
```

其中：

```math
h^{(\ell)}
=
\sigma(W^{(\ell)}h^{(\ell-1)} + b^{(\ell)})
```

这里：

- $x$ 是输入；
- $h^{(\ell)}$ 是第 $\ell$ 层 hidden representation；
- $W^{(\ell)}$ 和 $b^{(\ell)}$ 是 learnable parameters；
- $\sigma$ 是 [[Activation Function]]；
- $y$ 是输出。

## 🧱 Why “Deep”?

“Deep” 指的是网络有很多层。

浅层 neural network 可能只有一两个 hidden layers；  
deep neural network 有多层 hidden layers，因此可以学习更复杂的 hierarchical representations。

例如在图像任务中：

```math
\text{pixels}
\rightarrow
\text{edges}
\rightarrow
\text{shapes}
\rightarrow
\text{object parts}
\rightarrow
\text{objects}
```

在 language model 中：

```math
\text{tokens}
\rightarrow
\text{local patterns}
\rightarrow
\text{syntax}
\rightarrow
\text{semantics}
\rightarrow
\text{task-relevant representations}
```

>[!note]
>深度不是唯一重要因素。
>
>网络是否好用还取决于 architecture、training recipe、data、optimization 和 systems constraints。

## 🧩 Function Composition View

Deep neural network 可以看成多个函数的复合：

```math
f(x)
=
f_L
\circ
f_{L-1}
\circ
\cdots
\circ
f_1(x)
```

每一层都是一个 transformation：

```math
f_\ell(h)
=
\sigma(W_\ell h + b_\ell)
```

>[!note]
>深度带来的表达能力来自多层 nonlinear transformations 的组合。
>
>如果没有 activation function，很多线性层叠在一起仍然只是一个线性变换。

## 🔁 Training

DNN 的参数通过 training 学习。

基本过程是：

1. [[Forward Pass]]：输入经过网络得到 prediction；
2. 计算 [[Loss Function]]；
3. [[Backpropagation]]：计算 loss 对参数的 gradients；
4. [[Optimizer]]：根据 gradients 更新参数。

抽象写成：

```math
\theta
\leftarrow
\theta
-
\eta \nabla_\theta \mathcal{L}
```

其中：

- $\theta$ 是全部参数；
- $\eta$ 是 learning rate；
- $\mathcal{L}$ 是 loss。

## 🤖 Examples

常见 deep neural networks 包括：

- Convolutional Neural Network
- Recurrent Neural Network
- [[Transformer]]
- [[Feed-Forward Network]]

>[!note]
>[[Transformer]] 也是一种 deep neural network architecture。
>
>现代 [[Large Language Model (LLM)]] 通常是非常深的 Transformer-based neural network。

## 🚫 Common Confusions

### Deep Neural Network 不等于 Deep Learning

[[Deep Learning]] 是研究和使用 deep neural networks 的领域 / 方法范式。  
[[Deep Neural Network]] 是具体的模型结构。

### Deep 不只等于 layers 多

层数多是 deep 的基本特征，但模型能力还取决于：

- hidden dimension；
- activation function；
- residual connection；
- normalization；
- training data；
- optimizer；
- compute budget。

---

>[!summary] My Understanding
>[[Deep Neural Network]] 是包含多层 hidden layers 的 neural network。
>
>它通过多层 nonlinear transformations，把输入逐步转换成更抽象的 representations。
>
>DNN 的参数通过 [[Backpropagation]] 和 [[Optimizer]] 学习；现代 Transformer-based LLM 可以看作一种非常大的 deep neural network。

## 🔗 Connections

- [[Neural Network]]
- [[Deep Learning]]
- [[Feed-Forward Network]]
- [[Activation Function]]
- [[Backpropagation]]
- [[Optimizer]]
- [[Loss Function]]
- [[Transformer]]
- [[Large Language Model (LLM)]]