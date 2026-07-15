
#DeepLearning #NeuralNetwork #Transformer #Statistics #Regression

[Residual Connection](<./Residual%20Connection.md>) 里的 “residual” 和 regression 里的 “residual” 不是同一个概念。
它们都和“剩余部分 / 差值”有关，但在数学对象、作用位置和含义上完全不同。

## 🧠 Core Idea

>**Note**
>在 neural network 中，[Residual Connection](<./Residual%20Connection.md>) 指的是：
>
>
```math
>x_{\text{out}} = x + F(x)
>
```
>
>也就是保留输入 $x$，再加上一个 learned update $F(x)$。
>
>在 regression 中，residual 指的是：
>
>
```math
>r = y - \hat{y}
>
```
>
>也就是真实值和预测值之间的误差。

所以：

| Term | Formula | Meaning |
|---|---|---|
| [Residual Connection](<./Residual%20Connection.md>) | $x + F(x)$ | 网络层内部的信息流结构 |
| Regression residual | $y - \hat{y}$ | 模型预测和真实观测之间的误差 |

## 🔁 Residual Connection

在 neural network 中，residual connection 是一种 architecture design。

它的基本形式是：

```math
x_{\text{out}}
=
x
+
F(x)
```

其中：

- $x$ 是输入 representation；
- $F(x)$ 是某个 neural network sublayer 学到的 transformation；
- $x_{\text{out}}$ 是更新后的 representation。

>**Note**
>Residual connection 的核心直觉是：
>
>一层网络不需要从零生成一个全新的 representation，而只需要学习对原 representation 的增量修正。

也可以写成：

```math
x_{\text{out}} - x = F(x)
```

从这个角度看，$F(x)$ 是这一层学到的 update / delta。

在 [Transformer Block](<./Transformer%20Block.md>) 中，attention 和 MLP 通常都通过 residual connection 写回 [Residual Stream](<./Residual%20Stream.md>)：

```math
x'
=
x
+
\mathrm{Attention}(\mathrm{Norm}(x))
```

```math
x_{\text{out}}
=
x'
+
\mathrm{MLP}(\mathrm{Norm}(x'))
```

这里的 “residual” 强调的是：

- 保留原输入；
- 学习增量更新；
- 改善 gradient flow；
- 支持深层网络训练。

## 📉 Regression Residual

在 regression 中，residual 是一个统计建模概念。

给定观测值：

```math
y_i
```

和模型预测：

```math
\hat{y}_i
```

regression residual 定义为：

```math
r_i
=
y_i
-
\hat{y}_i
```

它表示模型没有解释掉的那部分误差。

如果写成线性回归：

```math
y = X\beta + \epsilon
```

拟合后得到：

```math
\hat{y} = X\hat{\beta}
```

则 residual vector 是：

```math
r
=
y
-
\hat{y}
```

>**Note**
>Regression residual 衡量的是 prediction error。
>
>它用于判断模型拟合得好不好，而不是 neural network 内部的数据流结构。

Regression residual 常用于：

- residual plot；
- checking model assumptions；
- diagnosing heteroskedasticity；
- measuring fit quality；
- deriving least squares properties。

## ⚖️ Main Difference

| Aspect | Residual Connection | Regression Residual |
|---|---|---|
| Field | deep learning / neural networks | statistics / regression |
| Object | hidden representation update | prediction error |
| Formula | $x + F(x)$ | $y - \hat{y}$ |
| Occurs where | inside network layers | after prediction |
| Purpose | stabilize deep networks, preserve information | measure unexplained error |
| Learned? | $F(x)$ is learned by network parameters | residual is computed after fitting |
| Used in Transformer? | yes | not as architecture concept |

>**Important**
>在 Transformer 里说 residual connection / residual stream 时，不是在说模型预测错了多少。
>
>它是在说 hidden state 如何通过加法结构在 network layers 中流动。

## 🧩 Why They Share the Word “Residual”

这两个概念都可以理解成某种 “剩余 / 差值”，但剩余的对象不同。

### In Regression

Regression residual 是：

```math
\text{actual output}
-
\text{predicted output}
```

也就是模型还没解释掉的误差：

```math
r = y - \hat{y}
```

### In Residual Network

Residual connection 可以从另一个角度写成：

```math
F(x)
=
x_{\text{out}}
-
x
```

也就是说，sub-layer 学的是从 $x$ 到 $x_{\text{out}}$ 的差值 / update：

```math
\text{new representation}
=
\text{old representation}
+
\text{learned update}
```

>**Note**
>所以 residual network 里的 residual 更接近 “learned correction / increment”。
>
>Regression residual 更接近 “unexplained error”。

## 🧱 In Transformer

在 [Transformer Block](<./Transformer%20Block.md>) 中，residual connection 的作用非常核心。

一个 Pre-Norm decoder-only block 可以写成：

```math
x'
=
x
+
\mathrm{Attention}(\mathrm{Norm}(x))
```

```math
x_{\text{out}}
=
x'
+
\mathrm{MLP}(\mathrm{Norm}(x'))
```

这里没有出现 regression residual：

```math
y - \hat{y}
```

出现的是 hidden states 的 update：

```math
x \rightarrow x + \Delta
```

其中 $\Delta$ 来自 attention 或 MLP。

>**Note**
>Transformer 中的 [Residual Stream](<./Residual%20Stream.md>) 就是这些 residual connections 串联后形成的主 hidden state 流。
>
>每一层都在已有 representation 上添加新的信息，而不是完全重写 representation。

## 🧠 Intuition

可以用两个问题区分它们：

### Residual Connection 问的是：

>**Question**
>这一层 neural network 如何更新 hidden representation？

答案是：

```math
x_{\text{out}} = x + F(x)
```

### Regression Residual 问的是：

>**Question**
>模型预测和真实值之间差多少？

答案是：

```math
r = y - \hat{y}
```

这两个问题不在同一个层面。

## 🚫 Common Confusion

>**Warning**
>不要把 Transformer 里的 residual stream 理解成“误差流”。
>
>[Residual Stream](<./Residual%20Stream.md>) 不是预测误差，也不是 loss 的 residual。
>
>它是 hidden representations 在 Transformer layers 中流动的主路径。

同样：

```math
\text{Residual stream}
\neq
y - \hat{y}
```

而是类似：

```math
X^{(0)}
\rightarrow
X^{(1)}
\rightarrow
\cdots
\rightarrow
X^{(L)}
```

其中每一步都通过：

```math
X^{(\ell+1)}
=
X^{(\ell)}
+
\Delta^{(\ell)}
```

进行更新。

## 📌 Minimal Distinction

>**Summary**
>一句话区分：
>
>[Residual Connection](<./Residual%20Connection.md>) 是网络内部的加法结构：
>
>
```math
>x + F(x)
>
```
>
>Regression residual 是预测之后的误差：
>
>
```math
>y - \hat{y}
>
```

---

>**Summary** — My Understanding
>[Residual Connection](<./Residual%20Connection.md>) 和 regression residual 都用了 “residual” 这个词，但它们不是同一个概念。
>
>在 Transformer 中，residual connection 表示保留当前 hidden state，并加上 attention 或 MLP 计算出的更新量。
>
>在 regression 中，residual 表示真实值和预测值之间的差。
>
>所以 Transformer 的 [Residual Stream](<./Residual%20Stream.md>) 不是误差流，而是 hidden representation 的主数据流。

## 🔗 Connections

- [Residual Connection](<./Residual%20Connection.md>)
- [Residual Stream](<./Residual%20Stream.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Transformer](<./Transformer.md>)
- Regression Analysis
- Least-squares problem]]