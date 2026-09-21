#Mathematics #Optimization #AI

## ✅ What is Stochastic Gradient Descent？

SGD 是用随机采样的 mini-batch gradient 估计 full gradient 的 Gradient Descent。

在第 $t$ 步，随机采样一个 mini-batch $B_t$：

```math
g_t
=
\frac{1}{|B_t|}
\sum_{i\in B_t}
\nabla \ell_i(\theta_t)
```

然后更新：

```math
\theta_{t+1}
=
\theta_t-\alpha_t g_t
```

- stochastic 来自每一步随机采样的 $B_t$；

>**Important** — 这个方向是对的吗？
> - $-g_t$ 是当前 mini-batch loss 的下降方向；
> - 但因为 $g_t$ 带有 sampling noise，它不保证每一步都让 full loss 下降。
> - 但 uniform sampling 下：
>```math
>  \mathbb E[-g_t\mid\theta_t]=-\nabla F(\theta_t)
>```
>$g_t$ 在 expectation 上等于 full gradient，所以平均来看，update direction 仍然朝向 full loss 下降的方向。

---
## For one step

1. 取出 mini-batch $B_t$
2. 这部分 mini-batch 作为输入放进 $\mathcal L_\theta(x)$ ，这里的参数是当前 step 的 $\theta_t$
3. 计算 [loss](<../Artificial%20Intelligence/Language%20modeling/02%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)
4. backward 得到 $g_t$
5. [Optimizer](<../Artificial%20Intelligence/Transformer/05%20-%20Training/Optimizer.md>) 更新参数，得到 $\theta_{t+1}$
6. 下一步换下一个 mini-batch...

---

## In LLM

SGD 每次只使用一个 mini-batch 完成一次 [optimizer step](<../Artificial%20Intelligence/Transformer/05%20-%20Training/Optimizer.md>)。batch size 越大，gradient estimate 通常越稳定，但每一步需要的 compute 和 memory 也越多。

>**Note**
> 数学中的 SGD 通常指上面的 update rule；`torch.optim.SGD` 是它的软件实现，还可以加入 momentum、Nesterov 和 weight decay。[AdamW](<AdamW.md>) 也使用 stochastic mini-batch gradient，但它是另一个 adaptive optimizer。
