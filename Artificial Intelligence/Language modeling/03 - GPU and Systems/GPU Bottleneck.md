#AI #LLM #LanguageModeling #GPU 

> **GPU compute 很强，但 memory bandwidth 跟不上**

所以要判断一个算子是 compute-bound 还是 [[GPU Memory Bound]]
这里涉及 [[Arithmetic Intensity]] 和 data movement

---
对于一个算子：

$$
I_{\text{arith}}
=
\frac{\text{FLOPs}}{\text{bytes moved}}
$$

这是 [[Arithmetic Intensity]]。

对于一个硬件：

$$
I_{\text{accel}}
=
\frac{\text{peak FLOP/s}}{\text{memory bandwidth}}
$$

这是 Accelerator Intensity。

判断规则：

$$
I_{\text{arith}} < I_{\text{accel}}
\Rightarrow
\text{memory-bound}
$$

$$
I_{\text{arith}} > I_{\text{accel}}
\Rightarrow
\text{compute-bound}
$$

>[!note]
> 这一步可以帮助判断优化方向：如果是 [[GPU Memory Bound]]，应优先减少 memory traffic；如果是 compute-bound，应优先提高计算单元利用率。
