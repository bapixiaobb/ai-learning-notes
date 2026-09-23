#LanguageModeling #Attention

Linear Attention 改变了 [standard softmax attention](<Self-Causal%20Attention.md>) 的数学定义，得到一种计算成本更低的 attention，它和 softmax attention 不是一个东西

## Why we need it?

[softmax attention computation cost](<Self-Causal%20Attention.md#computation-cost>) 引入了一个思考：**怎样设计一种相似度和归一化方式，让 attention 既保留“按匹配程度汇总信息”的含义，又允许这样的计算重排。**

经典线性注意力用 feature map 来构造这座桥。[Linear Attention 原始论文](https://proceedings.mlr.press/v119/katharopoulos20a.html)

## Intuition

> **Note** — softmax attention 里 softmax 做了三件事
>
> 1. 指数化：把 score 变成正数。
> 2. 归一化：让一个 query 对所有 keys 的权重之和为 1。
> 3. 规定权重如何分配：score 越大，得到的权重按指数关系增加。

所以这里有一个 trade-off: **为了更低的计算成本，我们接受怎样的匹配行为变化，以及可能的模型效果变化。** “是否允许负权重”只是其中一个设计选择；线性注意力完全可以使用非负权重。

## How we get it?

从 [Derivation Of Attention](<Derivation%20Of%20Attention.md>) 的式子出发：

```math
o_i=\frac{\sum_{j=1}\exp(S_{ij})V_{j,:}}{\sum_{j}\exp(S_{ij})}\quad\quad\quad\quad S_{ij}=\frac{q_i^Tk_j}{\sqrt{d_k}}
```
我们希望找到一种权重能够替代 $\exp(S_{ij})$ 把 query 和 key 的计算拆开

Matrix Multiplication and Basis 我们可以知道矩阵乘法本质上是一种 mapping，引入一个 feature map：
```math
\phi:\mathbb R^{d_k}\rightarrow\mathbb R^r
```
它把 query 和 key 映射到一个新的特征空间。在那里用内积衡量相似度：
```math
\operatorname{sim}(q,k)=\phi(q)^\top\phi(k)
```
这里 $\phi$ 本身可以是非线性的；$r$ 是映射后的特征维度，不一定等于 $d_k$。为了保留非负加权平均的含义，可以选择输出非负特征的 $\phi$，并保证归一化分母大于零。

这样得到：
```math
o(q)= \frac{ \sum_j\bigl(\phi(q)^\top\phi(k_j)\bigr)v_j }{ \sum_j\phi(q)^\top\phi(k_j) }
```
>**Important** — **关键变化是：query 和 key 分别完成变换，最后才通过内积相遇。**
>
>因此，对一个固定的 $q$，可以把 $\phi(q)$ 从求和中提出来。

定义：

```math
S=\sum_j\phi(k_j)v_j^\top \in\mathbb R^{r\times d_v}, \qquad z=\sum_j\phi(k_j) \in\mathbb R^r
```

那么输出的行向量就是：

```math
\boxed{ o(q)^\top = \frac{\phi(q)^\top S}{\phi(q)^\top z} }
```

**$S$ 和 $z$ 都只依赖 keys、values，不依赖当前 query。** 因此先把它们算好，所有 query 都能复用；每个 query 不再需要逐个扫描 $T$ 个 key。

如果令 $\phi(x)=x$，这里的 $S$ 就退回你刚才理解的 $K^\top V$。不过原始向量的内积可能为负，所以直接这样做不能保证非负权重——feature map 的选择也在决定相似度的性质。

## Choice of $\phi$

**随便选择一个 $\phi$，得到的是一种新的 attention，并不自动等于或近似于 softmax attention。**

$\phi$ 思路可以分成两条路线：

- **想近似 softmax**：使 $\phi(q)^\top\phi(k)$ 尽量接近原来的 $\exp(q^\top k/\sqrt{d_k})$，再按权重总和归一化。[经典线性注意力论文](https://proceedings.mlr.press/v119/katharopoulos20a.html)
- **想设计另一种好用的 attention**：不要求它接近 softmax，只要求这个匹配方式容易计算，并且模型训练后的效果足够好。

## Casual Linear Attention

```math
\boxed{o_t=\sum_{i\le t}v_i(k_i^\top q_t)= \left(\sum_{i\le t}v_i k_i^\top\right)q_t}
```
**这里 $q_t、k_i、v_i$ 都是列向量，$o_t$ 也是列向量**。所以 $k_i^\top q_t$ 是一个数，用它给整个 $v_i$ 加权。

- 不使用 [softmax](<Softmax.md>)：直接把内积 $k_i^\top q_t$ 当权重，没有指数，也没有除以权重总和。
- 保留 causal 限制：只加到 $i=t$，当前 token 只能使用自己和之前的 key/value。
