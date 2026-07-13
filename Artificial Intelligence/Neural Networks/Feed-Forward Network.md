#DeepLearning #NeuralNetwork #Transformer #Architecture

Feed-Forward Network 指信息只沿一个方向从输入流向输出的 neural network。  
它没有 recurrent connection，也不会把后面的输出再反馈回前面的层。

## 🧠 Core Idea

>**Note**
>Feed-forward 的意思是：
>
>```math
>\text{input}
>\rightarrow
>\text{hidden layers}
>\rightarrow
>\text{output}
>```
>
>信息只向前传播，不形成循环。

最简单的 feed-forward network 可以写成：

```math
h
=
\sigma(W_1 x + b_1)
```

```math
y
=
W_2 h + b_2
```

其中：

- $x$ 是输入；
- $W_1, W_2$ 是 learnable weights；
- $b_1, b_2$ 是 biases；
- $\sigma$ 是 [Activation Function](<./Activation%20Function.md>)；
- $y$ 是输出。

## 🧩 Relation to Neural Network

[Neural Network](<./Neural%20Network.md>) 是更大的概念。  
Feed-forward network 是 neural network 的一种基本结构。

可以理解成：

```math
\text{Feed-Forward Network}
\subset
\text{Neural Network}
```

它的特点是：

- 有 input layer、hidden layers、output layer；
- 每一层接收上一层的输出；
- 信息只向前流动；
- 参数通过 training 被学习；
- 通常用 [Backpropagation](<./Backpropagation.md>) 计算 gradients。

>**Note**
>你之前那张 [Neural Network](<./Neural%20Network.md>) 卡讲的是神经网络这个大类。
>
>Feed-Forward Network 这张卡更具体，关注的是一种没有循环结构、信息单向流动的网络结构。

## 🧱 Basic Structure

一个多层 feed-forward network 可以写成：

```math
h^{(1)}
=
\sigma(W^{(1)}x + b^{(1)})
```

```math
h^{(2)}
=
\sigma(W^{(2)}h^{(1)} + b^{(2)})
```

```math
\cdots
```

```math
y
=
W^{(L)}h^{(L-1)} + b^{(L)}
```

也就是：

```math
x
\rightarrow
h^{(1)}
\rightarrow
h^{(2)}
\rightarrow
\cdots
\rightarrow
y
```

>**Note**
>如果 feed-forward network 有很多 hidden layers，它就可以是 [Deep Neural Network](<./Deep%20Neural%20Network.md>) 的一种。

## ⚖️ Feed-Forward vs Recurrent

| Network Type | Information Flow | Example |
|---|---|---|
| Feed-Forward Network | input → output，单向无循环 | MLP |
| Recurrent Neural Network | hidden state 会循环传递 | RNN, LSTM |
| [Transformer](<../Transformer/Transformer.md>) | 通过 attention 在 sequence positions 间交互 | Self-Attention |

>**Important**
>Feed-forward network 没有 memory state。
>
>它不会像 RNN 那样把前一时刻的 hidden state 递归传到下一时刻。

## 🧮 Feed-Forward Network in Transformer

在 [Transformer Block](<../Transformer/Transformer%20Block.md>) 中，Feed-Forward Network 通常也叫 [MLP](<../Transformer/MLP.md>)。

它对每个 token position 独立作用：

```math
x_t
\in
\mathbb{R}^{d_{\text{model}}}
```

```math
\mathrm{FFN}(x_t)
=
W_2 \sigma(W_1 x_t + b_1) + b_2
```

常见 shape 是：

```math
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
```

对于整个 sequence：

```math
X \in \mathbb{R}^{T \times d_{\text{model}}}
```

FFN 输出仍然是：

```math
\mathrm{FFN}(X)
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

>**Note**
>Transformer 里的 FFN 不负责 token mixing。
>
>它对每个 token position 独立做 nonlinear transformation。
>
>Token positions 之间的信息交换主要由 [Self-Attention](<../Transformer/Self-Attention.md>) 完成。

## 🔍 Intuition

在 Transformer block 中，可以粗略理解为：

```text
Self-Attention: tokens 之间交换信息
Feed-Forward Network: 每个 token 自己内部做非线性处理
```

也就是说，attention 先把上下文信息混进每个 token representation，FFN 再对这个 representation 做进一步加工。

## **🦙 Modern LLM Variant**

在 [Original Transformer](<../Transformer/Original%20Transformer.md>) 中，FFN 常用 [ReLU](<../Transformer/ReLU.md>)

>**Note**
所以在 LLM 语境里，[Feed-Forward Network](<./Feed-Forward%20Network.md>)、[MLP](<../Transformer/MLP.md>)、FFN 经常指 Transformer block 中 attention 后面的 per-token nonlinear module。

## **🚫 Common Confusions**

### **1. Feed-forward 不等于 forward pass**

[Forward Pass in Transformer](<../Transformer/Forward%20Pass%20in%20Transformer.md>) 指一次模型前向计算过程。  
[Feed-Forward Network](<./Feed-Forward%20Network.md>) 指一种 neural network structure。

它们名字相似，但不是同一个概念。

### **2. Feed-forward 不等于 Transformer 整体**

Transformer 里面有 FFN / MLP，但 Transformer 不只是 FFN。  
Transformer 还包括：

- [Self-Attention](<../Transformer/Self-Attention.md>)
- [Residual Connection](<../Transformer/Residual%20Connection.md>)
- [Normalization](<../Transformer/Normalization.md>)
- [Positional Encoding](<../Transformer/Positional%20Encoding.md>)

### **3. FFN 不做 token mixing**

在 Transformer 中，FFN 对每个 token 独立作用。  
真正让 token positions 之间交换信息的是 [Self-Attention](<../Transformer/Self-Attention.md>)。

---

>**Summary** — My Understanding
> Feed-Forward Network 是信息单向流动的 neural network。
>
>在普通 neural network 语境下，它指从 input 到 hidden layers 再到 output 的前馈结构。
>
>在 Transformer 语境下，FFN 通常指每个 [Transformer Block](<../Transformer/Transformer%20Block.md>) 里的 per-token MLP：
>
>```math
d_{\text{model}}  
\rightarrow  
d_{\text{ff}}  
\rightarrow  
d_{\text{model}}  
>```
>
>它不负责 token mixing，而是对每个 token representation 做 nonlinear processing。

## **🔗 Connections**

- [Neural Network](<./Neural%20Network.md>)
- [Deep Neural Network](<./Deep%20Neural%20Network.md>)
- [MLP](<../Transformer/MLP.md>)
- [Transformer Block](<../Transformer/Transformer%20Block.md>)
- [Self-Attention](<../Transformer/Self-Attention.md>)
- [Activation Function](<./Activation%20Function.md>)
- [ReLU](<../Transformer/ReLU.md>)
- [Forward Pass in Transformer](<../Transformer/Forward%20Pass%20in%20Transformer.md>)
- [Backpropagation](<./Backpropagation.md>)