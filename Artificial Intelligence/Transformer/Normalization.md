#Mathematics #Statistics #MachineLearning #DeepLearning

Normalization] 指把数据或向量按照某种规则重新缩放，使它们落在更稳定、更可比较的尺度上。

它的核心目的通常是：**控制 scale**。

## 🧠 Core Idea

>[!note]
>Normalization 的核心思想是：
>
>把原始数值转换到一个更标准、更稳定的尺度上。
>
>它通常不会改变数据之间的主要结构关系，而是改变数值的 scale / location。

很多机器学习方法对 scale 很敏感。  
例如，如果一个 feature 的数值范围是：

$$
[0, 100000]
$$

另一个 feature 的范围是：

$$
[0, 1]
$$

那么模型或优化算法可能会过度受到大尺度 feature 的影响。

Normalization 试图让不同维度、样本或向量处在更可比较的尺度上。

## 📏 What Is Being Normalized?

Normalization 需要先明确：**对谁做 normalization？**

常见对象包括：

| Object | Example |
|---|---|
| scalar feature | 对一个 feature column 标准化 |
| vector | 把一个向量缩放到 unit norm |
| matrix | 对每一列、每一行或整个矩阵 normalization |
| activation | 对 neural network hidden states normalization |
| probability vector | 让元素非负且总和为 1 |

>[!important]
>Normalization 不是一个唯一公式。
>
>不同场景下，normalize 的对象不同，公式也不同。

## ⚖️ Standardization / Z-score Normalization

最常见的 normalization 之一是 standardization，也叫 z-score normalization。

给定数据：

$$
x
$$

均值为：

$$
\mu
$$

标准差为：

$$
\sigma
$$

则标准化为：

$$
z
=
\frac{x - \mu}{\sigma}
$$

标准化后通常有：

$$
\text{mean} = 0
$$

$$
\text{standard deviation} = 1
$$

>[!note]
>Z-score normalization 同时做了两件事：
>
>- center：减去 mean；
>- scale：除以 standard deviation。

## 📐 Min-Max Normalization

Min-max normalization 把数值缩放到固定区间，例如 $[0,1]$。

给定：

$$
x_{\min}
=
\min_i x_i
$$

$$
x_{\max}
=
\max_i x_i
$$

则：

$$
x'
=
\frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

这样：

$$
x' \in [0,1]
$$

>[!note]
>Min-max normalization 保留相对顺序，但对 outliers 比较敏感。

## 🧭 Vector Normalization

有时 normalization 指把一个向量缩放到固定 norm。

例如 L2 normalization：

$$
\hat{x}
=
\frac{x}{\|x\|_2}
$$

其中：

$$
\|x\|_2
=
\sqrt{\sum_i x_i^2}
$$

这样：

$$
\|\hat{x}\|_2 = 1
$$

>[!note]
>L2 normalization 主要控制向量长度。
>
>它保留方向，但去掉 scale。

也可以使用 L1 normalization：

$$
\hat{x}
=
\frac{x}{\|x\|_1}
$$

其中：

$$
\|x\|_1
=
\sum_i |x_i|
$$

## 🎲 Probability Normalization

在概率中，normalization 常指把一组非负数缩放成概率分布。

给定 scores：

$$
a_1, a_2, \dots, a_n
$$

如果 $a_i \geq 0$，可以定义：

$$
p_i
=
\frac{a_i}{\sum_{j=1}^{n} a_j}
$$

于是：

$$
\sum_{i=1}^{n} p_i = 1
$$

>[!note]
>这类 normalization 的目标是让总质量为 1。
>
>概率分布必须满足 non-negative 和 sum-to-one。

[[Softmax]] 也可以看作一种把 arbitrary logits 转成 probability distribution 的 normalization：

$$
p_i
=
\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

## 🧮 Normalization in Optimization

Normalization 在 optimization 中很重要，因为 scale 会影响 gradient-based methods。

如果不同 features 的 scale 差异很大，loss landscape 可能变得 badly conditioned。

直觉上，optimization path 会更难走：

$$
\text{different scales}
\Rightarrow
\text{ill-conditioned geometry}
\Rightarrow
\text{slower optimization}
$$

>[!note]
>Normalization 往往能改善 numerical stability 和 optimization dynamics。
>
>它不是改变问题本质，而是让问题在数值上更容易处理。

这和 [[Condition number]]、[[Gradient Descent]]、[[An Optimization Problem]] 有自然联系。

## 🧠 Normalization in Neural Network

在 neural networks 中，normalization 通常用于控制 activations 的尺度。

常见方法包括：

- [[Layer Normalization]]
- [[RMSNorm]]

它们的区别主要在于：

>[!question]
>mean / variance 是沿着哪些 dimensions 计算的？

例如：

- BatchNorm 通常跨 batch dimension 统计；
- LayerNorm 通常对一个样本的 feature dimension 统计；
- RMSNorm 只根据 root mean square 缩放，不 subtract mean。

>[!note]
>在 neural networks 中，normalization 的目的通常是让 activations 的尺度更稳定，从而让 training 更稳定。

## 🔍 Centering vs Scaling

Normalization 经常包含两类操作：

### Centering

减去某个 reference value，例如 mean：

$$
x - \mu
$$

作用是改变 location。

### Scaling

除以某个 scale，例如 standard deviation 或 norm：

$$
\frac{x}{s}
$$

作用是改变 magnitude。

>[!important]
>不是所有 normalization 都包含 centering。
>
>例如 [[Layer Normalization]] 会 subtract mean，但 [[RMSNorm]] 不 subtract mean，只做 scaling。

## 🧩 Why Normalization Matters

Normalization 的作用可以总结为：

- 控制 scale；
- 提高 numerical stability；
- 让不同 features / activations 更可比较；
- 改善 optimization geometry；
- 减少某些维度因数值过大而主导计算；
- 在概率中保证总质量为 1。

>[!note]
>Normalization 本质上是在选择一个更合适的数值尺度。
>
>它不一定改变信息内容，但会改变计算过程中的数值表现。

## 🚫 Common Confusions

### Normalization 不等于 regularization

Normalization 控制数值尺度；[[Regularization]] 控制模型复杂度或限制过拟合。

例如：

$$
\frac{x - \mu}{\sigma}
$$

是 normalization。

而：

$$
\mathcal{L} + \lambda \|\theta\|_2^2
$$

是 regularization。

### Normalization 不一定让数据服从正态分布

“normalization” 这个词容易让人误以为是变成 normal distribution。  
但大多数 normalization 只是 rescaling，不保证数据变成 Gaussian。

### Normalization 不一定是除以最大值

除以最大值只是某一种 normalization。  
更常见的还有 z-score、min-max、L2 normalization、[[Softmax]] normalization 等。

---

>[!summary] My Understanding
>Normalization] 是一种控制 scale 的数学方法。
>
>它把数据、向量、矩阵、activations 或 scores 按某种规则重新缩放，使数值更稳定、更可比较。
>
>不同场景下 normalization 的公式不同：z-score normalization 控制 mean 和 variance，min-max normalization 控制取值区间，L2 normalization 控制向量长度，[[Softmax]] normalization 把 logits 转成概率分布。
>
>理解 normalization 的关键是先问：
>
>**我在 normalize 什么对象？沿着什么维度 normalize？想控制什么 scale？**

## 🔗 Connections
- [[Layer Normalization]]
- [[RMSNorm]]
- [[Condition number]]