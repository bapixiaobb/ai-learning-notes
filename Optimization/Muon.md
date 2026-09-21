#AI #LanguageModeling

Momentum Orthogonalized by Newton-Schulz: [Muon 原始算法说明](https://kellerjordan.github.io/posts/muon/)

通常只处理 Transformer/MLP 的二维 hidden weight matrices
- [attention](<../Artificial%20Intelligence/Transformer/02%20-%20Attention/Query%20Key%20Value.md>) 的 $W_Q,W_K,W_V,W_O$；
- MLP 的 $W_{\text{up}},W_{\text{gate}},W_{\text{down}}$。
```math
\theta = \theta_{\text{Muon}} \cup \theta_{\text{AdamW}}
```
经验上这些通常仍使用 AdamW：
- token embedding；
- final LM head；
- RMSNorm/LayerNorm weights；
- bias；
- 其他 scalar/vector parameters。

Emerging [Optimizer](<../Artificial%20Intelligence/Transformer/05%20-%20Training/Optimizer.md>)

```math
\theta_{t+1}=\theta_t-\alpha_t \operatorname{Ortho}(M_t)
```
上式只表示 Muon 的核心 update。实际实现通常还会独立应用 [Weight Decay](<../Artificial%20Intelligence/Language%20modeling/01%20-%20System/Weight%20Decay.md>)：
```math
W_{t+1}=(1-\alpha_t\lambda)W_t-\alpha_t s(W)\operatorname{Ortho}(M_t)
```
其中 $s(W)$ 是 [shape scaling](<Muon%20Scaling.md>)。
## Difference from AdamW

它改变了传统 [AdamW](<AdamW.md>) 按元素（element-wise）计算缩放的思路，转而利用 Matrix Orthogonalization 来引导参数更新

对于一个二维 parameter：$W\in\mathbb{R}^{m\times n}$，它对应的：

```math
G_t=\frac{\partial L}{\partial W}=\begin{bmatrix} \cdots & \cdots\\ \cdots & \cdots \end{bmatrix},\,\,\, m_t,\,\,\, v_t \in\mathbb{R}^{m\times n}
```

不过它们虽然“长得是矩阵”，AdamW 并不使用它们的 matrix structure：
```math
p_t^{\text{AdamW}} = -\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

AdamW 中的所有计算都是 element-wise。Muon 则真的使用 matrix structure：

```math
G_t \longrightarrow M_t\quad\text{momentum matrix} \longrightarrow \operatorname{Ortho}(M_t)
```

Muon 描述了矩阵的整体结构
>**Question** — 整个矩阵主要想沿哪些方向变化？能不能让不同方向的更新更均衡？

得到：

```math
P_t^{\text{Muon}} = -\operatorname{Ortho}(M_t)
```

---
## Momentum

AdamW 的 $m_t$ 和 Muon 的基础 momentum buffer 本质上是同一组历史 gradients 的指数加权和，只差 normalization factor。
#### Momentum buffer

得到 $G_t$ 后，保存历史方向
```math
B_t=\mu B_{t-1}+G_t
```
这里 $B_t\approx \text{AdamW 的 }m_t$，只差缩放 ($m_t=(1-\mu)B_t$)

#### Nesterov (choice)
但是 Muon 默认还会多做一步：
```math
M_t=G_t+\mu B_t
```
在历史趋势上，再额外强调一次当前 gradient $G_t$。

公式展开：
```math
B_t = G_t+\mu G_{t-1}+\mu^2G_{t-2}+\cdots
```
而：
```math
M_t = (1+\mu)G_t +\mu^2G_{t-1} +\mu^3G_{t-2} +\cdots
```
所以 Nesterov 版本对最新的 $G_t$ 更敏感，不会完全被过去的 momentum 拖着走。

>**Note** — Muon 通常没有 [bias correction](<AdamW%20Early-Step%20Estimation%20Bias.md>)：正交化根本不在乎 scale
>```math
>\operatorname{Ortho}(cM) = \operatorname{Ortho}(M), \qquad c>0
>```

## Matrix Orthogonalization

主要区别
```math
\boxed{ \text{AdamW: } m_t \longrightarrow \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} }
```
```math
\boxed{ \text{Muon: } B_t \longrightarrow M_t \longrightarrow \operatorname{Ortho}(M_t) }
```

AdamW 中用：
```math
-\frac{1}{\sqrt{\hat v_t}+\epsilon}
```
来做 element-wise second-moment scaling

Muon 是直接用 $M_t$ 的正交化来代替

>**Note** — **Target:**
> 把 $M$ 变成与它最接近的 semi- orthogonal matrix，也就是 Orthogonal Procrustes Problem；
>```math
>O^* = \arg\min_{O^\top O=I} \|O-M\|_F
>```
> 而 semi-orthogonal constraint 要求 singular values 等于 1，所以最终自然丢掉了原来的 singular-value magnitudes。
>
> 这个问题的最优解是 $M$ 的 polar factor (tall matrix)：
>```math
> \boxed{ M(M^\top M)^{-1/2} = \operatorname{polar}(M) = UV^\top }
>```

#### Muon 想计算：
```math
\operatorname{Ortho}(M)=\operatorname{polar}(M)
```
>**Note** — 数学对应是：
>
>对于一个 scalar：$\frac{g}{|g|}=\operatorname{sign}(g)$
>对于一个 matrix：$M(M^\top M)^{-1/2}=UV^\top$
>所以 Muon 的 orthogonalization，可以看成是把 scalar 的“除以自己的大小”推广到了整个 matrix。

直接计算 SVD 或 matrix inverse square root 比较昂贵，所以 Muon 使用 Newton-Schulz 迭代近似得到这个结果

首先会做一个 Normalization 找一个合适的初始值（N-S converge）:
```math
X_0 = \frac{M}{\|M\|_F+\epsilon}
```
然后丢进 N-S 里（这里用 N-S 很重要的一个理由是能用 bf16）
![Muon](<attachments/Muon.png>)
## Scale Mismatch

>**Note** — orthogonalization 可能会在长方形矩阵上发生“尺度失真”

$\operatorname{Ortho}(M_t)$ 的 Frobenius norm 是固定的，令 $O_t=\operatorname{Ortho}(M_t)$
```math
\boxed{ \|O_t\|_F=\sqrt{\min(m,n)} }
```
这个矩阵的 RMS 是
```math
\text{RMS}(O_t)=\frac{1}{\sqrt{\max(m,n)}}
```
如果 Muon 直接更新：
```math
\theta_{t+1}=\theta_t-\alpha O_t
```
那么 parameter update 的 RMS 是：
```math
\operatorname{RMS}(\Delta W) = \frac{\alpha}{\sqrt{\max(m,n)}}
```
这意味着即使使用同一个 learning rate $\alpha$：

- 小 matrix 的每个 parameter 走得比较远；
- 大 matrix 的每个 parameter 走得比较近；
- 不同 shape 的 matrices 获得不同的 update scale。

所以这里我们要用 [Muon Scaling](<Muon%20Scaling.md>) 来消除 shape dependency

---
## [Quantization](<../Artificial%20Intelligence/Quantization/Quantization.md>)

相比于 [AdamW](<../Artificial%20Intelligence/Quantization/Optimizer%20Quantization.md>) 来说，Muon 通常能减少 persistent optimizer-state memory，但 Newton–Schulz 会增加临时 peak memory

>**Important** — **My understanding**
> AdamW 产生的 matrix update 可能主要集中在少数 singular directions。Muon 把非零 singular values 拉到接近相同尺度，使较弱的 directions 也能获得明显更新。这被认为是 Muon 提高训练效率的可能原因之一，但它是机制解释和经验观察。
