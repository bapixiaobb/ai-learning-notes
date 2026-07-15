#AI #LanguageModeling #LLM

一种 [GPU Memory Bound](<../03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>) 的优化方法：**用低精度格式参与实际计算**
# Why it is an optimization for [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>)
## Memory Optimization

这是一个很直接的减少数据搬运的方法，既然有 [GPU Memory Bound](<../03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>)，那我就**每个数字少用一点 bit 表示。** 这个方法叫做 [Quantization](<../../Transformer/Quantization.md>)

>**Note** — if you have fewer bits, you have fewer bits to move。

```
FP32 → BF16 / FP16 → FP8 → FP4
```

精度越来越低，但：
- 每个数占的 bytes 更少；
- memory traffic 更少；
- tensor core / matrix unit 的吞吐更高；
- 代价是数值误差、overflow/underflow、training instability。

>**Example**
>一个 FP32 数字：`32 bits = 4 bytes`
>一个 BF16 / FP16 数字：`16 bits = 2 bytes`
>一个 FP8 数字：`8 bits = 1 byte`
>如果一个 vector operation 用 FP32，读一个数 4 bytes，写一个数 4 bytes，总共 8 bytes memory access；如果精度减半，搬运的 bytes 也跟着减半。

>**Note** — FP32 → BF16
>这个比较特殊不像是 FP8 这种 bit 位差别很大的数据类型转换
>一般是 cast to BF16，不是 [Quantization](<../../Transformer/Quantization.md>)，BF16 自己有 exponent 和 mantissa，不需要额外 scale，然后模型直接用 BF16 做训练。
## Compute Optimization

**专门硬件对低精度矩阵乘法更快。**

比如 NVIDIA Tensor Core 对 BF16、FP16、FP8 这些格式有专门支持。

低精度除了减少 memory bandwidth 需求之外，compute 也有收益；只是因为还要 quantize / dequantize，所以实际收益会被稀释。

---
# Which operation should use which number format

>**Note** — Quantization $\neq$ 把所有 FP32 都转成 INT8 / FP8。
>有些操作可以低精度，有些操作必须高精度
>low precision training 的难点就在判断哪些操作应该用什么数据类型

1. MatMul 通常适合低精度：因为它是大头计算，而且 tensor core 对它支持很好
	1. weights: low precision
	2. [activations](<../../Neural%20Networks/Activations.md>): low precision
	3. matmul: tensor core 快速计算
	4. accumulation: higher precision
2. Softmax / exp / normalization 这些操作可能对数值范围、稳定性更敏感。
	- 所以这些操作经常要保留 `FP32` 或者至少 `BF16`

---
# Low precision of MatMul

>**Important** — **乘法输入可以低精度，但累加通常需要更高精度。**
>

所以矩阵乘法 $A\times B = C$ 一般是：
```
A: low precision
B: low precision
multiply: low precision hardware
accumulate: higher precision
C: maybe FP32 / BF16 / FP8
```

Why? 因为矩阵乘法里一个元素是：`C[i, j] = sum_k A[i, k] B[k, j]`，C 的一个元素有很多项相加，如果每一步累加都用很低精度，误差会不断积累。

---
>**Question** — low precision 讲成减少 [GPU Memory Bound](<../03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>)，那它对 compute 有没有帮助？
>有，compute 肯定也有帮助。
>对 quantized numbers 做乘法，基本可以得到接近线性的 compute improvement。
>但是因为还要 quantize / dequantize，所以整体收益会被 diluted
>
>**为什么收益不是 2x / 4x？**
>因为系统还要做：
>```
>quantize
>dequantize
>scale factor 读取
>scale factor 应用
>可能还要存 transpose copy
>某些 layer 不能 quantize
>```
>所以最终 end-to-end 可能只有：$20\% - 30\%$，这种实际收益，而不是理论上的 2x。
