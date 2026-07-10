#AI #Statistics #LanguageModeling 

是一种在 [[Deep Learning]] ([[Neural Network]]) 中常用的 [[Activation Function]]

>[!note]
> 它能把一个包含任意实数的向量（通常是 [[Neural Network]] 最后一层输出 score，也叫 Logits）映射为一个概率分布


# Mathematical Definition

给定 logits，其中 $K$ 是类别总数：

```math
z = (z_1, z_2, \dots, z_K)
```

那么对于第 $k$ 个类别的 Softmax 输出值 $\hat{p}_h$ 定义如下：

```math
\hat{p}_k
=
\frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
```

# Intuition

Softmax 计算可以分为三步：
1. **Exponential**：把所有预测得分 $z_i$ 转化为正数。
	-  得分越高，指数爆炸后的值就越大
	- 目的是，拉大差距，softmax 有强分类器放大效果
2. **Summation**：计算所有类别指数化后的总和
3. **Normalization**：用每个类别的指数值除以总和

# Safe Softmax

在计算机工程实现中，为了防止 overflow 必须要做硬件保护措施

利用 softmax 的平移不变性，会减去 $\max(x)$ 

```math
\mathrm{softmax}(x_i)=\frac{e^{x_i-\max(x)}}{\sum_je^{x_j-\max(x)}}
```
[[Flash Attention]] 里为了配合 tile by tile 计算，有 [[Online Softmax]] 