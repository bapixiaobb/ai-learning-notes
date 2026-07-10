#Systems #NumericalLinearAlgebra #LanguageModeling

>[!note]
> **FLOPs** 是 **floating-point operations**，表示浮点运算次数。
>
> 它衡量的是一个计算任务总共需要做多少次浮点计算。

## 与 FLOP/s 的区别

>[!note]
> **FLOPs** 衡量总计算量；**FLOP/s** 衡量计算速度。
>
> 两者的关系类似于：
>
> - distance：总路程
> - speed：速度

| 概念 | 含义 |
|---|---|
| FLOPs | total amount of computation |
| FLOP/s | floating-point operations per second |

如果一个任务需要：$10^{23}$ FLOPs，而硬件实际速度是：$10^{18} \text{ FLOP/s}$

那么理想耗时约为：
```math
\frac{10^{23}}{10^{18}}
=
10^5
\text{ seconds}
```

## 在 Language Model 中

>[!note]
> 在 language model training 中，FLOPs 用来估算训练总计算量。
>
> 模型越大、训练 tokens 越多，总 FLOPs 越高。

常见近似：

```math
\text{Training FLOPs} \approx 6ND
```

其中：

- $N$：number of model parameters
- $D$：number of training tokens

## 与 Matrix Multiplication 的关系

>[!note]
> Transformer 中大部分计算来自 matrix multiplication。
>
> 因此估算 Transformer FLOPs 的基础通常是 Matrix Multiplication FLOPs。

对于：

```math
A \in \mathbb{R}^{m \times k},
\quad
B \in \mathbb{R}^{k \times n}
```

矩阵乘法：
```math
C = AB
```

大约需要：

```math
2mnk
```

FLOPs。

## 为什么重要

>[!note]
> FLOPs 可以帮助估算训练成本、训练时间和模型规模是否可行。
>
> 但 FLOPs 只描述“理论计算量”，不能单独说明实际训练速度。

实际速度还取决于：

- FLOP per Second
- Memory Bandwidth
- [[Model FLOPs Utilization]]
- parallelism
- communication overhead
- kernel efficiency

## Related
- [[Training Compute - 6ND]]
- [[Resource Accounting]]
- [[Model FLOPs Utilization]]