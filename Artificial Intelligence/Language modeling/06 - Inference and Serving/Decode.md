#AI #LanguageModeling

是 [Inference](<Inference.md>) 里 [Prefill](<Prefill.md>) 之后逐步产生新 tokens 的过程

## Decode 的起点
假设 prompt 是：
```math
 [x_1,x_2,x_3,x_4]
```
模型先处理整段 prompt，得到：

```
logits: [batch, 4, vocab_size]
```

只使用最后一个位置：$z_4$，经过 softmax 得到：
```math
 p(x_5\mid x_1,x_2,x_3,x_4)
```
然后通过 greedy、[temperature](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md#temperature-scaling>)、[top-p](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md#top-p-sampling>) 等 decoding strategy，选择或采样：
```math
 x_5
```
这个 $x_5$ 就是第一个 generated token。

## Sequential

得到 $x_5$ 后，它要加入 context：
```math
 [x_1,x_2,x_3,x_4,x_5]
```
模型接下来计算：
```math
 p(x_6\mid x_1,x_2,x_3,x_4,x_5)
```
采样出 $x_6$，再继续：
```math
 p(x_7\mid x_1,\dots,x_6)
```
所以
```
先得到 xₜ₊₁
→ 把 xₜ₊₁ 加入 context
→ 才能生成 xₜ₊₂
```
因此不同 Decode steps 之间存在 sequential dependency。

## Sequence length

从上面的推导可以看出，Decode 过程中[输入](<Inference.md#basic-llm-inference-flow>)的长度为
```math
\boxed{T=1}
```

一般 $S$ 很大，这就导致了读取很大的 $W$ 但是只给一个新 token 做计算

- 读取 K/V 的数据量：$O(S)$；
- attention 计算量：也是 $O(S)$。

**arithmetic intensity 很糟糕**

推导见 [Arithmetic Intensity in Inference](<Arithmetic%20Intensity%20in%20Inference.md>)

## Decode 优化

侧重节省仿存，例如：[Reduce KV cache size](<Reduce%20KV%20cache%20size.md>)、KV cache 量化、[Operator fusion](<../../GPU%20and%20NPU/Operator%20fusion.md>) 减少中间结果搬运、更高带宽利用

Decode 慢的主要原因是带宽喂不饱算力，给 decode 堆算力通常换不来等比例加速，必须从访存入手
