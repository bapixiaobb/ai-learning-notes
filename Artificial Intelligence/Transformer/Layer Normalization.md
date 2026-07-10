#DeepLearning #NeuralNetwork #Normalization #Transformer

[[Layer Normalization]] 是一种 normalization 方法。  
它对每个 token 的 hidden vector 在 feature dimension 上做 normalization。

## 🧠 Core Idea

给定一个 hidden vector：

$$
x \in \mathbb{R}^{d}
$$

LayerNorm 先计算 mean：

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i
$$

再计算 variance：

$$
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2
$$

然后 normalize：

$$
\mathrm{LayerNorm}(x)
=
\gamma \odot
\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
+
\beta
$$

其中：

- $\gamma$ 是 learnable scale；
- $\beta$ 是 learnable bias；
- $\epsilon$ 是防止除零的小常数；
- $\odot$ 是 element-wise multiplication。

## 📌 In Transformer

在 Transformer 中，LayerNorm 通常作用在每个 token 的 hidden representation 上：

$$
x_t \in \mathbb{R}^{d_{\text{model}}}
$$

也就是说，它不会跨 token 做 normalization，而是对单个 token vector 的 feature dimension 做 normalization。

>[!note]
>LayerNorm 用来稳定 hidden states 的尺度，使 deep Transformer 更容易训练。

## ⚖️ LayerNorm vs RMSNorm

LayerNorm 会做两件事：

1. subtract mean；
2. divide by standard deviation。

也就是：

$$
x
\rightarrow
\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

而 [[RMSNorm]] 不 subtract mean，只根据 root mean square 缩放。

---

>[!summary] My Understanding
>[[Layer Normalization]] 对每个 token 的 hidden vector 做 normalization。
>
>它会减去 mean，再除以 standard deviation，使 hidden vector 的尺度更稳定。

## 🔗 Connections

- [[Normalization]] 
- [[RMSNorm]]
- [[Transformer Block]]
- [[Pre-Norm Transformer]]
- [[Post-Norm Transformer]]