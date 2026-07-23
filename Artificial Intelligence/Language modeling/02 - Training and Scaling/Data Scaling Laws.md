#AI #LanguageModeling

Data scaling law 是 [Scaling Law](<Scaling%20Law.md>) 在 data axis 上的具体形式：在 model class 和 training setup 近似固定时，研究 dataset size $D$ 增加后 held-out loss / error 如何变化。

Simple formula that maps dataset size (n) to error

理想化的完整 learning curve：随着训练数据量增加，模型的 generalization error 通常会经历“几乎学不到 → power-law 下降 → 接近极限”的三个区域。
![expected data scaling law.png](<../attachments/expected%20data%20scaling%20law.png>)
## Data scaling laws for [language models](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20%28LLM%29.md>)

上面那张图中间的区域叫 Power-law Region

核心就是这个 fit
```math
L(D)=L_\infty +AD^{-\alpha}
```
$D$ 是 dataset size / training tokens；在 log-log 图中（就是下图）：
```math
\log (L-L_\infty)=-\alpha \log D+\log A
```

**Loss and dataset size is linear on a log-log plot** （上图中间区域）
![data scaling law.png](<../attachments/data%20scaling%20law.png>)
which means estimation error decay polynomial as dataset size increases.

这里有两个决定性的因素：
- slope $-\alpha$：增加数据时，error 下降多快；
- intercept $\log A$：这批数据从什么水平开始，或者说数据有多“有效”。

这里的 slope 只有在 Power-law Region 区间内才能外推
## Dataset Size

 忽略 $L_\infty$
```math
E(n)\propto n^{-\alpha}
```
取对数：
```math
\log E(n)=-\alpha\log n+C
```

因此：

- Mean estimation / classical linear regression：常见 $E(n)\sim n^{-1}$，log-log slope 是 $-1$；
- neural scaling law：大约 $E(n)\sim n^{-0.1}$ 到 $n^{-0.3}$，slope 大约是 $-0.1$ 到 $-0.3$。
## Data mixture：affects the offset, not the slope.

假设有两种训练数据：
- Wikipedia
- News
不同 mixture 可能产生近似平行的 scaling curves：
![Data composition.png](<../attachments/Data%20composition.png>)较好的 mixture 把整条 loss 曲线向下移动，但是增加数据时的下降速度没有明显改变。

因此，如果不同 mixture 的 slope 真相同：
> **Important** — 小模型上最好的 mixture，很可能在大模型上仍然最好。所以实践中有时候不必拟合复杂的 mixture scaling law

## Data repetition：训练 token 不等于新信息

> **Question** — In practice, we have finite data – how does repeating examples affect scaling?

假设你只有 100B unique tokens，却训练了四个 epoch：$\text{processed tokens}=400B$
- 前几个 epoch，重复数据可能还不会明显伤害；
- 重复越来越多以后，每次重复的边际价值下降；
- 实际 loss 会落后于“全部都是 fresh data”时的 scaling-law 预测。
此时的曲线可能变平：
![repeated data in scaling laws.png](<../attachments/repeated%20data%20in%20scaling%20laws.png>)

> **Important** — 干预经常只改变 intercept？
> 更好的 data mixture, regularization, ensemble, 更好的数据过滤。这些操作可能让相同数据量下的 loss 更低，也就是整条曲线下移；但它们不一定改变 exponent $\alpha$。


**很多 engineering improvement 改变的是 scaling curve 的 intercept，而不是 slope**
