#DeepLearning #NeuralNetwork #Transformer #LanguageModeling

MLP 是 multilayer perceptron。
在 Transformer block 中，MLP 对每个 token position 独立做 nonlinear transformation。

## 🧠 Core Idea

>**Note**
>在 Transformer 中：
>
>- [Self-Attention](<./Self-Attention.md>) 负责 token positions 之间的信息交换；
>- MLP 负责对每个 token representation 做 nonlinear processing。

给定一个 token hidden vector：

```math
x_t \in \mathbb{R}^{d_{\text{model}}}
```

普通 MLP 可以写成：

```math
\mathrm{MLP}(x_t)
=
W_2 \sigma(W_1 x_t + b_1) + b_2
```

其中：

- $W_1$ 把 hidden dimension 扩大；
- $\sigma$ 是 activation function；
- $W_2$ 把维度映射回 $d_{\text{model}}$。

## 📐 Shape

Transformer 中的 MLP 通常是：

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

MLP 输出仍然是：

```math
\mathrm{MLP}(X)
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

>**Important**
>MLP 不改变 sequence length。
>
>它对每个 token position 独立作用，不负责 token mixing。

## 🔍 Intuition

可以把 Transformer block 中的 MLP 理解为：

```text
对每个 token 的 hidden vector 做一个小型 neural network transformation
```

Self-attention 已经把上下文信息混进了 token representation。
MLP 接着对这个 representation 做 nonlinear feature transformation。
## **💻 Minimal Implementation**

```python
import torch
import torch.nn as nn

class TransformerMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        x: (B, T, d_model)
        """
        return self.fc2(self.act(self.fc1(x)))
```

## **🦙 SwiGLU Variant**

Modern LLM 中常见的是 [SwiGLU](<./SwiGLU.md>) MLP。

---

>**Summary** — My Understanding
>MLP 在 Transformer block 中负责 per-token nonlinear transformation。
>
>它不做 token mixing，而是对每个 token 的 hidden vector 独立作用。
>
>常见 shape 是：
>
>
```math
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
>
```

## **🔗 Connections**

- [Transformer Block](<./Transformer%20Block.md>)
- [Feed-Forward Network](<../Neural%20Networks/Feed-Forward%20Network.md>)
- [Activation Function](<../Neural%20Networks/Activation%20Function.md>)
- GELU
- [SwiGLU](<./SwiGLU.md>)
- [Residual Connection](<./Residual%20Connection.md>)
- [Residual Stream](<./Residual%20Stream.md>)
- [Model Architecture](<../Language%20modeling/05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>)