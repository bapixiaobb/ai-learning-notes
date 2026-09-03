#GPU

```math
d=a\times b+c
```
# Fused？

```
普通 mul + add：
t = round(a × b)
d = round(t + c)       // 两次舍入

FMA：
d = round(a × b + c)   // 中间结果不舍入，只有一次舍入
```

硬件上来讲，精度会高一些（昇腾有专门的指令做 $\times +$ )

# 为什么 MatMul 本质上是一堆 FMA？
比如：

$$ C=A B $$

其中一个输出元素：

$$ C_{ij}=\sum_k A_{ik}B_{kj} $$

展开就是：

$$ C_{ij} = A_{i0}B_{0j} + A_{i1}B_{1j} + A_{i2}B_{2j} +\cdots $$
程序可以想成：

```
float acc = 0;

for (int k = 0; k < K; k++) {
    acc = fma(A[i][k], B[k][j], acc);
}
```

所以一个 Thread 如果负责一个 `C[i][j]`，它实际上就在不停做：

```
acc = a0 * b0 + acc
acc = a1 * b1 + acc
acc = a2 * b2 + acc
...
```

也就是很多次 **scalar FMA**。
