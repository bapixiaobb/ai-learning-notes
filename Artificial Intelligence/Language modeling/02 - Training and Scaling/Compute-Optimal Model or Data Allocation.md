#AI #LanguageModeling

[Chinchilla paper](<Chinchilla%20Scaling%20Law.md>) 用来估计 [compute-optimal model/data allocation](<Scaling%20Law.md>) 的三种方法

(Chinchilla 把三种方法系统地放在一起交叉验证)

>**Question** — 给定 Compute $C$，$N$ 取多大，$D$ 取多少，loss 最低

## Method 1：Minimum over runs

训练许多不同大小的模型，得到很多条：$\text{loss versus FLOPs}$ 训练曲线（这个方法 Kaplan 也部分使用过）

![minimum over runs](<../../attachments/minimum%20over%20runs.png>)

对于每一个 compute budget $C$，在所有曲线中寻找最低的 loss：

> 花这么多 compute 时，哪一个模型取得了最好结果？

把所有最低点连接起来，就是所有 training curves 的 lower envelope。

然后记录每个最低点对应的：

- model size $N$；
- training tokens $D$；
- compute $C$。

再拟合：

```math
N_{\text{opt}}(C),\qquad D_{\text{opt}}(C)
```

Chinchilla 用这个方法预测 Gopher 的 compute budget 应该训练大约 **67B parameters** 的模型。
## Method 2：IsoFLOP

先固定一个 [compute budget](<Training%20Compute%20-%206ND.md>)：$C\approx6ND$ 然后尝试不同的 model size $N$。为了保持 compute 不变，$N$ 改变时必须相应改变 $D$：
![IsoFLOPS](<../../attachments/IsoFLOPS.png>)
- 模型变大 → tokens 减少；
- 模型变小 → tokens 增加。

把这些实验的最终 loss 画出来，通常得到一条 U-shaped curve：

- 左边：模型太小，data 再多也吸收不了；
- 右边：模型很大，但训练 tokens 太少，undertrained；
- 最低点：这个 compute budget 下最好的 $N,D$ 分配。

然后换几个不同的 compute budget，重复这个过程，再拟合所有最低点如何随 compute 变化。

它预测 Gopher 的 compute budget 应该训练约 **63B parameters**，与 Method 1 的 67B 很接近。

## Method 3：Joint fit (ChinChilla fit had problem)

先假设一个 joint scaling law：

```math
L(N,D) = E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}
```

然后训练许多不同的 $(N,D)$ 组合，用 least squares 把整张 loss surface 拟合出来。
![joint fit](<../../attachments/joint%20fit.png>)
得到这张地形图后，再求解：

```math
\min_{N,D}L(N,D) \qquad \text{subject to } C\approx6ND
```

也就是：

> 在固定 compute 的曲线上，寻找 loss surface 最低的位置。

这个方法使用的 joint functional form 来自 Rosenfeld/Kaplan 一类工作，也不是 Chinchilla 完全新发明的。原论文的 Method 3 给出大约 **40B parameters**，与前两种方法有些差异；

#### 这个 method 3 的原始 fitting 可能存在问题

后来有人重新拟合，Method 3 也更接近 Method 1、Method 2 和 20 tokens/parameter。
