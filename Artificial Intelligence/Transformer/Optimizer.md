#AI #LanguageModeling

Mathematical foundation see: An Optimization Problem

In [training](Training vs Inference) process, after [Forward Propagation](<../Neural%20Networks/Forward%20Propagation.md>) (see [TransformerLM.jpeg](<./TransformerLM.jpeg>)) we will get a loss function computed by [Cross Entropy Loss](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>).

>**Important** — We need to define the optimizer to minimize this loss

# The SGD Optimizer

The simplest gradient-based optimizer is Stochastic Gradient Descent (SGD). We start with randomly initialized parameters $\theta_0$. Then for each step $𝑡 = 0,\cdots,T-1$,  we perform the following update:
```math
\theta_{t+1}\leftarrow\theta_t-\alpha_t \nabla L(\theta_t;B_t)
```

where  $B_t$ is a random batch of data sampled from the dataset $𝐷$, and the learning rate $\alpha_t$ and batch size $|B_t|$ are hyperparameters.

---
# AdamW

Typical applications set $(\beta_1,\beta_2)$ to $(0.9, 0.999)$, but large language models like LLaMA (H. Touvron et al., 2023) and GPT-3 (T. B. Brown et al., 2020) are often trained with $(0.9, 0.95)$

![adamw.png](<./adamw.png>)

AdamW 仍然是一阶方法，只使用当前 mini-batch gradient：
```
gₜ = ∇L(θₜ; Bₜ)
```

它比 SGD 多做三件事：

### 1. 平滑 gradient 的方向：First Moment

方向的移动平均： $m_t = β_1m_{t-1} + (1-β_1)g_t$

$m$ 是 gradients 的 exponential moving average，类似 momentum。**$m$ 不是 gradients，而是历史 gradient 的指数滑动平均**

- $g_t$：当前这一刻的力
- $m_t$：过去很多步累积出来的运动趋势

>**Example** — Momentum
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

AdamW 还会单独执行：$\theta \leftarrow \theta - \alpha\lambda \theta$ ，它会把参数轻微拉向零。

>**Important** — $W$ 的关键就在这里：weight decay 与 gradient update 解耦。

---
# How to find step size??

In training [Transformer](<./Transformer.md>) step size is called learning rate, we have [Learning Rate Schedule](<./Learning%20Rate%20Schedule.md>)