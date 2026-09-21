#AI #MachineLearning

```math
\boxed{ \theta_{t+1} = \theta_t-\alpha_t\,s(W)\,O_t }
```
## Background

[Muon 对于长方形矩阵会出现 scale mismatch 的情况](<Muon.md#scale-mismatch>)

在传统的 AdamW 或 SGD 中，矩阵变大，梯度的“总能量”（norm）也会自然变大。

>**Important** — 但 **[Muon](<Muon.md>) 做了正交化**。这一步正交化带来了一个非常关键的物理限制：
> **正交化后，整个更新矩阵 $O$ 的“总能量”（所有元素的平方和），被严格锁定为了矩阵的【短边长】 $\min(M, N)$。**
>- 如果矩阵是 $1000 \times 1000$（正方形）：总能量上限是 $1000$。
>- 如果矩阵是 $1000 \times 8000$（长方形）：总能量上限**依然只有 $1000$**（因为短边是 1000）。

**长方形矩阵里的参数，每次更新走出的 Step Size（[Learning Rate](<../Artificial%20Intelligence/Transformer/05%20-%20Training/Learning%20Rate%20Schedule.md>)）比正方形矩阵里的参数小得多**。在训练过程中，这些长方形层就会表现为“根本学不动”、“更新速率极慢”，拉低整网的收敛速度。

## `AdjustLR`（Scaling）[PyTorch Muon 文档](https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html)

**根据矩阵的具体形状，按比例把 $\alpha$ 补放大！**
```math
\text{AdjustLR}(\alpha; \text{shape}(\theta)) = \alpha \times \text{补偿系数}
```

- **补偿的原理**：长边越长（比如 $N=8000$），参数被摊薄得越厉害，$\text{AdjustLR}$ 就会算出一个更大的乘法因子（通常正比于 $\sqrt{\frac{N}{M}}$ 或类似形状比例），强行把场景 B 的学习率放大。
- **最终目的**：确保无论这个 Parameter Tensor 是正方形、长方形还是极扁的扁平矩阵，**每一个参数在单步更新中所感受到的实际步长尺度（Scale）是一致的（Consistently）**。

## Moonshot 的 `match_rms_adamw` ：$s(W)$

我们希望消除：

```math
\frac{1}{\sqrt{\max(m,n)}}
```

带来的 shape dependency，所以乘回：

```math
\sqrt{\max(m,n)}
```

即：

```math
\Delta \theta = -\alpha \sqrt{\max(m,n)} O_t
```

此时：

```math
\operatorname{RMS} \left( \sqrt{\max(m,n)}O_t \right) = 1
```

如果想让目标 RMS 不是 1，而是 \(c\)，就写成：

```math
s(\theta)=c\sqrt{\max(m,n)}
```

Moonshot 为了匹配典型 AdamW update 的 RMS，经验上取：

```math
c=0.2
```

于是：

```math
\boxed{ s(\theta)=0.2\sqrt{\max(m,n)} }
```
