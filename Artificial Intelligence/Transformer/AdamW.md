#AI #optimizer

理论上 Hessian 可以根据不同方向的曲率调整更新，但代价太高；AdamW 不计算 Hessian，而是用历史 gradients 的统计量做便宜、可扩展的 coordinate-wise scaling。

## ✅ What is AdamW？

AdamW 是 [Transformer](<./Transformer.md>) / large language model training 中常用的 first-order adaptive [optimizer](<./Optimizer.md>)。

它和 [SGD](<./Stochastic%20Gradient%20Descent.md>) 一样从 mini-batch gradient 出发，但会用 gradient history 平滑 noisy direction，并对每个 parameter coordinate 调整 effective step size。

> **Important**
> AdamW = Adam + decoupled weight decay。**W** 指的是 weight decay 与 adaptive gradient update 解耦。

## 它比 SGD 多做三件事：

### 1. 平滑 gradient 的方向：First Moment

方向的移动平均： $m_t = β_1m_{t-1} + (1-β_1)g_t$

$m$ 是 gradients 的 exponential moving average，类似 momentum。**$m$ 不是 gradients，而是历史 gradient 的指数滑动平均**

- $g_t$：当前这一刻的力
- $m_t$：过去很多步累积出来的运动趋势

> **Example — Momentum**
> 可以类比 momentum 吧，因为物理动量里，不只有大小还有方向。
>
> 比如一个球一直往右滚，它有“往右的动量”。你稍微给它一个往左的小力，它不会立刻往左飞，而是先被减速。
> 这就是“惯性”的感觉：过去一直往右 → 现在倾向于继续往右

### 2. 根据每个坐标的历史 gradient 大小调整有效 step size：Second Moment

gradient 大小的移动平均：$v_t = β_2v_{t-1} + (1-β_2)g_t^2$

$v$ 记录每个 parameter element 最近的 gradient magnitude。

最终方向会除以：$\sqrt{v} + \varepsilon$

- 某坐标的 gradient 长期很大 → $v$ 很大 → denominator 很大 → 该坐标有效 step size 变小
- 某坐标的 gradient 长期较小 → $v$ 较小 → denominator 较小 → 该坐标相对获得更大的有效 step size

### 3. 独立进行 weight decay

AdamW 还会单独执行：$\theta \leftarrow \theta - \alpha\lambda \theta$ ，它会把参数轻微拉向零。这里是给 optimization 加入 small-norm preference

> **Important** — $W$ 的关键就在这里：weight decay 与 gradient update 解耦。

这里有一个 trade-off：

```math
\text{low training loss} \quad\leftrightarrow\quad \text{small parameter norm}
```

因此，有限 $\lambda$ 下，AdamW 可能接受稍高一点的 training loss，换取小很多的 parameter norm。

## Pseudocode

Typical applications set $(\beta_1,\beta_2)$ to $(0.9, 0.999)$, but large language models like LLaMA (H. Touvron et al., 2023) and GPT-3 (T. B. Brown et al., 2020) are often trained with $(0.9, 0.95)$

![AdamW pseudocode](<./adamw.png>)

## [Quantization](<./Quantization.md>)

AdamW 的 $m$ 和 $v$ 一般会保留 FP32 accumulator

> **Note** — AdamW 是一个反复积累微小变化的 dynamical system。低精度的数值网格太粗，微小 update 可能直接被 rounding 掉，而且误差会进入后面所有 optimizer steps。

- $m_t = β_1m_{t-1} + (1-β_1)g_t$  $\beta_1=0.9$
  - 当前 gradient 对 $m_t$ 的贡献只有 $0.1g_t$，$g_t$ 小 -> $0.1g_t$ 更小
  - 低精度下，这个 gradient contribution 会被 rounding，$m_t$ 没有真正吸收当前 gradient
- $v_t = β_2v_{t-1} + (1-β_2)g_t^2$
  - 更敏感，累积 $g_t^2$
- 并且这些 optimizer state 是递归状态，当前累积的 error 会进入后面所有的 updates

> **Important**
> FP32 是 optimizer states 的稳健默认，不是数学上的强制要求。8-bit / low-precision optimizer 也可以使用，但通常需要 blockwise quantization、scaling 或 stochastic rounding 来保护小 updates。

$m$ 和 $v$ 都与 parameters shape 相同；如果使用 FP32，二者合计需要 $8$ bytes per parameter。因此它们是 [Resource Accounting](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 中重要的 memory cost，也是 [ZeRO](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/ZeRO.md>) Stage 1 首先 sharding 的对象。
