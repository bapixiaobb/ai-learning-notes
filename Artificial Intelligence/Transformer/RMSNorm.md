#DeepLearning #NeuralNetwork #Normalization #Transformer #LLM

[RMSNorm](<./RMSNorm.md>) 是 Root Mean Square Layer Normalization。
它和 [Layer Normalization](<./Layer%20Normalization.md>) 类似，但不 subtract mean，只根据 root mean square 缩放 hidden vector。

## 🧠 Core Idea

给定 hidden vector：

```math
x \in \mathbb{R}^{d}
```

RMSNorm 计算 root mean square：

```math
\mathrm{RMS}(x)
=
\sqrt{
\frac{1}{d}
\sum_{i=1}^{d} x_i^2
+
\epsilon
}
```

然后 normalize：

```math
\mathrm{RMSNorm}(x)
=
\gamma
\odot
\frac{x}{\mathrm{RMS}(x)}
```

其中：

- $\gamma$ 是 learnable scale；
- $\epsilon$ 是防止除零的小常数；
- $\odot$ 是 element-wise multiplication。

## 📌 In Transformer

在 modern decoder-only LLM 中，RMSNorm 常和 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) 搭配使用：

```math
x_{\text{out}}
=
x
+
F(\mathrm{RMSNorm}(x))
```

例如在 [Llama-style Architecture](<./Llama-style%20Architecture.md>) 中：

```math
x'
=
x
+
\mathrm{Attention}(\mathrm{RMSNorm}(x))
```

```math
x_{\text{out}}
=
x'
+
\mathrm{MLP}(\mathrm{RMSNorm}(x'))
```

## ⚖️ RMSNorm vs LayerNorm

| Method | Subtract Mean? | Divide by Scale? | Learnable Scale? |
|---|---|---|---|
| [Layer Normalization](<./Layer%20Normalization.md>) | Yes | standard deviation | Yes |
| [RMSNorm](<./RMSNorm.md>) | No | root mean square | Yes |

>**Note**
>RMSNorm 可以理解成 LayerNorm 的简化版本。
>
>它不做 mean-centering，只控制 hidden vector 的整体尺度。

---

>**Summary** — My Understanding
>[RMSNorm](<./RMSNorm.md>) 只用 root mean square 来缩放 hidden vector。
>
>它比 [Layer Normalization](<./Layer%20Normalization.md>) 少了 subtract mean 这一步，因此计算更简单，也常见于 modern LLM architecture。

## 🔗 Connections

- [Normalization](<./Normalization.md>)
- [Layer Normalization](<./Layer%20Normalization.md>)
- [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Llama-style Architecture](<./Llama-style%20Architecture.md>)