#DeepLearning #NeuralNetwork #Normalization #Transformer #LLM

[[RMSNorm]] 是 Root Mean Square Layer Normalization。  
它和 [[Layer Normalization]] 类似，但不 subtract mean，只根据 root mean square 缩放 hidden vector。

## 🧠 Core Idea

给定 hidden vector：

$$
x \in \mathbb{R}^{d}
$$

RMSNorm 计算 root mean square：

$$
\mathrm{RMS}(x)
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d} x_i^2
+
\epsilon
}
$$

然后 normalize：

$$
\mathrm{RMSNorm}(x)
=
\gamma
\odot
\frac{x}{\mathrm{RMS}(x)}
$$

其中：

- $\gamma$ 是 learnable scale；
- $\epsilon$ 是防止除零的小常数；
- $\odot$ 是 element-wise multiplication。

## 📌 In Transformer

在 modern decoder-only LLM 中，RMSNorm 常和 [[Pre-Norm Transformer]] 搭配使用：

$$
x_{\text{out}}
=
x
+
F(\mathrm{RMSNorm}(x))
$$

例如在 [[Llama-style Architecture]] 中：

$$
x'
=
x
+
\mathrm{Attention}(\mathrm{RMSNorm}(x))
$$

$$
x_{\text{out}}
=
x'
+
\mathrm{MLP}(\mathrm{RMSNorm}(x'))
$$

## ⚖️ RMSNorm vs LayerNorm

| Method | Subtract Mean? | Divide by Scale? | Learnable Scale? |
|---|---|---|---|
| [[Layer Normalization]] | Yes | standard deviation | Yes |
| [[RMSNorm]] | No | root mean square | Yes |

>[!note]
>RMSNorm 可以理解成 LayerNorm 的简化版本。
>
>它不做 mean-centering，只控制 hidden vector 的整体尺度。

---

>[!summary] My Understanding
>[[RMSNorm]] 只用 root mean square 来缩放 hidden vector。
>
>它比 [[Layer Normalization]] 少了 subtract mean 这一步，因此计算更简单，也常见于 modern LLM architecture。

## 🔗 Connections

- [[Normalization]]
- [[Layer Normalization]]
- [[Pre-Norm Transformer]]
- [[Transformer Block]]
- [[Llama-style Architecture]]