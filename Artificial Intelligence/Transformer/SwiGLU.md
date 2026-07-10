SwiGLU 是 [[Transformer]] FFN 里的 gated [[Activation Function]]

它的形式是：

```math
\mathrm{SwiGLU}(x) = \left(xW_1\right) \odot \mathrm{SiLU} \left(xW_2\right)
```
再输出投影：
```math
\mathrm{{FNN(x)=SwiGLU(x)W_3}}
```

本质是 [[GLU]] + [[SiLU]] 的变体，用来替代传统 [[ReLU]] / GeLU

- $x$ 是 hidden state
- $W_1$, $W_2$ 线性投影 / weight
- $SiLU$ [[SiLU]]
- $\odot$ 逐元素乘（gate 机制）
- $W_1$ : content branch
- $W_2$ : gate branch
- $W_3$ : output projection

## shape
- Batch size = $B$
- sequence length = $T$ (token 数)
- hidden size = $d$
- FFN 中间维度 = $d_{ff}$

**输入**
- $x$ : ($B$, $T$, $d$)
第一层（split 成两路）
- $W_1$ : ($d$, $d_{ff}$)
- $W_2$ : ($d$, $d_{ff}$)
得到
- $a = xW_1\rightarrow$ ($B$, $T$, $d_{ff}$)
- $b = xW_2\rightarrow$ ($B$, $T$, $d_{ff}$)
gate（逐元素）
- $g = a\odot \mathrm{SiLU}(b)\rightarrow$ ($B$, $T$, $d_{ff}$)
输出 projection
- $W_3$ : ($d_{ff}$, $d$)
- $\text{output}=gW_3\rightarrow$ ($B$, $T$, $d$)