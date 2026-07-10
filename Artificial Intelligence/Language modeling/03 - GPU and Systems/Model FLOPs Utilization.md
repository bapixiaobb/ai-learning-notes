
#DeepLearning #Systems #FLOPs

>[!note]
> **MFU** 是 **Model FLOPs Utilization**，表示模型训练时实际使用了硬件理论峰值算力的比例。
>
> 它用来衡量 deep learning training system 的计算效率。

## 基本形式

$$
\text{MFU}
=
\frac{\text{actual model FLOP/s}}{\text{hardware peak FLOP/s}}
$$

其中：

- **actual model FLOP/s**：训练过程中模型实际完成的有效计算速度
- **hardware peak FLOP/s**：硬件理论峰值计算速度

## 直觉

>[!note]
> MFU 回答的问题是：硬件理论上可以这么快，但训练这个模型时我们实际用了多少？
>
> 如果 MFU 很低，说明 [[GPU]] /TPU 的计算单元没有被充分利用。（50%以上算不错的）

例如：

$$
\text{hardware peak} = 1000 \text{ TFLOP/s}
$$

实际训练达到：

$$
400 \text{ TFLOP/s}
$$

那么：

$$
\text{MFU}
=
\frac{400}{1000}
=
40\%
$$

## 为什么重要

>[!note]
> 大模型训练不仅关心总 FLOPs，还关心这些 FLOPs 能不能高效地跑在硬件上。
>
> 同样的训练计算量，如果 MFU 更高，训练时间和成本就更低。

MFU 可以帮助判断：

- hardware utilization 是否足够高
- training throughput 是否合理
- bottleneck 是否来自 memory、communication 或 kernel inefficiency
- parallelism strategy 是否有效

## 与 FLOPs 的关系

>[!note]
> [[FLOPs]] 表示总计算量；MFU 表示这些计算在硬件上执行得有多高效。

如果训练总计算量是：

$$
C \text{ FLOPs}
$$

实际模型计算速度是：

$$
R \text{ FLOP/s}
$$

则理想化训练时间约为：

$$
\frac{C}{R}
$$

其中：

$$
R
=
\text{MFU}
\times
\text{hardware peak FLOP/s}
$$

所以：

$$
\text{training time}
\approx
\frac{C}
{\text{MFU} \times \text{hardware peak FLOP/s}}
$$

## 为什么 MFU 会低

>[!note]
> MFU 低通常不是因为模型“计算量不够”，而是因为系统中有其他瓶颈导致计算单元等待。

常见原因包括：

- Memory Bandwidth 不足
- communication overhead 太大
- batch size 或 microbatch size 太小
- kernel launch overhead
- tensor shape 不适合硬件
- activation checkpointing 带来的 recomputation
- data loading bottleneck
- parallelism strategy 不合理

## 与 [[Arithmetic Intensity]] 的关系

>[!note]
> 如果一个 workload 的 Arithmetic Intensity 太低，它可能是 [[GPU Memory Bound]]，导致硬件计算单元吃不满，从而降低 MFU。

判断 bottleneck 时常比较：

$$
I_{\text{arith}}
=
\frac{\text{FLOPs}}{\text{bytes moved}}
$$

和：

$$
I_{\text{accel}}
=
\frac{\text{peak FLOP/s}}{\text{memory bandwidth}}
$$

如果：

$$
I_{\text{arith}} < I_{\text{accel}}
$$

则 workload 更可能 [[GPU Memory Bound]]，MFU 往往较低。

## 与 Resource Accounting 的关系

>[!note]
> 在 [[Resource Accounting]] 中，MFU 用来把理论 compute estimate 转换成实际训练时间估计。
>
> 仅知道训练需要多少 FLOPs 不够，还需要知道系统能以多高的有效 FLOP/s 执行这些计算。

例如：

$$
\text{Training FLOPs} \approx 6ND
$$

但实际训练时间还取决于：

$$
\text{MFU}
$$

和硬件峰值性能。

## Related

- [[Resource Accounting]]
- [[FLOPs]]
- [[Training Compute - 6ND]]
- [[Deep Learning Systems]]