#LanguageModeling #Attention

从 [Self-Causal Attention](<Self-Causal%20Attention.md>) 看一下这个 Attention 是怎么算的

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+\text{mask}
\right)V
```

推到的 shape 从这里来：![TransformerLM.jpeg](<../../attachments/TransformerLM.jpeg>)
## [Q, K, V](<Query%20Key%20Value.md>)

先来看这个几个矩阵的格式

```math
Q=\begin{bmatrix}
\text{query}_1\\
\text{query}_2 \\
\vdots \\
\text{query}_S
\end{bmatrix}\quad \quad \quad \quad \quad \quad K=\begin{bmatrix}
\text{key}_1\\
\text{key}_2 \\
\vdots \\
\text{key}_S
\end{bmatrix}\quad \quad \quad \quad \quad \quad V=\begin{bmatrix}
\text{value}_1\\
\text{value}_2 \\
\vdots \\
\text{value}_S
\end{bmatrix}
```

```
Q：我要找什么
K：我可以通过什么被找到
V：找到我以后，可以拿走什么内容
```

## $\text{Score}=QK^T$

```math
\text{Score}=QK^\top=
\begin{bmatrix}
q_1^\top k_1 & q_1^\top k_2 & \cdots & q_1^\top k_S\\
q_2^\top k_1 & q_2^\top k_2 & \cdots & q_2^\top k_S\\
\vdots & \vdots & \ddots & \vdots\\
q_S^\top k_1 & q_S^\top k_2 & \cdots & q_S^\top k_S
\end{bmatrix}
```
| Score 维度 | 语义 |
| --- | --- |
| 第 `i` 行 | 哪个 query token 正在读取信息 |
| 第 `j` 列 | 它正在考虑哪个 source/key token |
| `Score[i,j]` | token `i` 与 token `j` 的匹配程度 |
```math
S_{ij}=\frac{q_i^Tk_j}{\sqrt{d_k}}
```
## [Softmax](<Softmax.md>)

softmax 是沿着每一行的方向进行的

```math
A=\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)=\begin{bmatrix}
A_1\\
A_2 \\
\vdots \\
A_S
\end{bmatrix}\begin{matrix} \Rightarrow \text{第 2 个 query token要怎样在所有 source token 之间分配注意力}\\ \\ \end{matrix}
```

这里忽略 mask

```math
A_{ij}
=
\frac{\exp(S_{ij})}
{\sum_{k=1}^{S}\exp(S_{ik})}
```

## $V$
```math
A=\begin{bmatrix}
A_1\\
A_2 \\
\vdots \\
A_S
\end{bmatrix}\in\mathbb{R}^{S\times S},
\qquad
V=
\begin{bmatrix}
v_1^\top\\
v_2^\top\\
\vdots\\
v_S^\top
\end{bmatrix}
\in\mathbb{R}^{S\times d_v}
```
```math
O=AV
=
\begin{bmatrix}
A_1V\\
A_2V\\
\vdots\\
A_SV
\end{bmatrix}
=
\begin{bmatrix}
o_1^\top\\
o_2^\top\\
\vdots\\
o_S^\top
\end{bmatrix}
\in\mathbb{R}^{S\times d_v}
```
- $A_i\in\mathbb R^{1\times S}$：第 $i$ 个 query 对所有 source 的权重；
- $V\in\mathbb R^{S\times d_v}$；
- $A_iV=o_i^\top\in\mathbb R^{1\times d_v}$。

把 $A_{ij}$ 带入

```math
O_{ia}
=
\sum_{j=1}^{S}A_{ij}V_{ja}
=
\frac{
\sum_{j=1}^{S}\exp(S_{ij})V_{ja}
}{
\sum_{k=1}^{S}\exp(S_{ik})
}\quad\quad\quad\quad o_i
=
\sum_{j=1}^{S}A_{ij}v_j
=
\frac{
\sum_{j=1}^{S}\exp(S_{ij})v_j
}{
\sum_{k=1}^{S}\exp(S_{ik})
}
```
