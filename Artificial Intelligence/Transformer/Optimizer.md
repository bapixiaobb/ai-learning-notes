#AI #LanguageModeling

Mathematical foundation see: An Optimization Problem

在一次 [training](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Training%20vs%20Inference.md>) step 中：

1. [model forward](<../Neural%20Networks/Forward%20Propagation.md>) 先得到 logits；
2. [Cross Entropy Loss](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>) 根据 logits 和 targets 计算 loss；
3. [Backpropagation](<../Neural%20Networks/Backpropagation.md>) 计算 gradients；
4. optimizer 使用 gradients 更新 parameters。

> **Important**
> Optimizer 负责根据 gradients 更新 parameters；它不负责计算 model output、loss 或 gradients。

---
## From Optimization Problem to Optimizer

从 An Optimization Problem 的角度，language model training 可以写成：

```math
\min_{\theta} F(\theta),
\qquad
F(\theta)=\mathbb{E}_{x\sim\mathcal D}[\ell(\theta;x)]
```

其中 $\theta$ 是 model parameters，$\ell(\theta;x)$ 是一个 training example 上的 [loss](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)。在有限 training set 上，也可以写成：

```math
F(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell(\theta;x_i)
```

An Optimization Problem 描述的是“要解什么问题”；optimizer 描述的是“如何迭代地更新 $\theta$ 来求解这个问题”。

许多 iterative optimization methods 都可以写成：

```math
\theta_{t+1}=\theta_t+\alpha_t p_t
```

- $p_t$ 是 direction，决定往哪里走；
- $\alpha_t$ 是 step size，在 deep learning 中通常称为 learning rate，决定走多远；

---
## First-Order Approximation

> **Note** — 从 Optimization 的角度，方向可以只利用 gradient，也可以进一步利用 Hessian 描述 curvature。

但是 language modeling 参数规模太大，完整 Hessian 的计算、存储和求解都不现实，loss 更是 stochastic、non-convex 的，Hessian 还可能 indefinite，所以实际训练主要使用 SGD、AdamW 这类基于 gradient 的一阶方法。

> **Note**
>> gradient：函数在当前位置的**局部斜率 / 一阶近似**；
>> Hessian：斜率如何变化，也就是**局部曲率 / 二阶近似**。

---
## 往哪走？

> **Question** — 哪个方向是 $F$  下降的方向？
>**方向导数：** $\nabla F(\theta_t)^T p_t$  --- $F$ 沿 $p_t$ 的方向导数
>
>当它小于 $0$ 时，沿 $p_t$ 走一小步会让 $F$ 下降，所以 $p_t$ 是一个 descent direction。

更一般地，可以先构造 local quadratic model，再生成：

```math
p_t=-B_t^{-1}\nabla F(\theta_t)
```

选择 $B_t=I$ 得到 Gradient Descent，选择 $B_t=\nabla^2F(\theta_t)$ 得到 Newton，使用 Hessian approximation 则得到 quasi-Newton。完整推导见 Descent Direction。

最直接的选择是 $p_t=-\nabla F(\theta_t)$，这就是 Gradient Descent。在 [Machine Learning](<../Fundamentals/Machine%20Learning.md>) 中，标准 Gradient Descent 每一步使用整个 dataset 计算 full gradient。

#### SGD

但在 large-scale training 中，计算 full gradient 的 cost 太高，所以通常使用随机采样的 mini-batch gradient 来估计它，这就是 [Stochastic Gradient Descent](<Stochastic%20Gradient%20Descent.md>)。

SGD 直接用 mini-batch gradient 会有两个问题：
- mini-batch gradient 有 noises，不同批次的 mini-batch 下降的方向可能完全相反
- SGD 用的同一个 learning rate，这会导致：
	- 如果方向很陡峭，稍微走多一点就会震荡
	- 如果方向很平缓，走得很慢
#### AdamW

因此 large language model training 通常使用 [AdamW](<AdamW.md>)：用 first moment 平滑 direction，用 second moment 对每个 parameter 做 adaptive scaling，并将 weight decay 与 gradient update 解耦。

---
## 走多远？

在 [Transformer](<Transformer.md>) training 中，step size 通常称为 learning rate。整个 training run 只创建一个 optimizer object，但它保存的 current learning rate 可以在每次 optimizer step 前被修改：

    step_size = get_lr_cosine_schedule(iteration, ...)
    optimizer.param_groups[0]["lr"] = step_size
    optimizer.step()

因此第 $t$ 次 optimizer step 使用的是：

```math
\alpha_t=\operatorname{schedule}(t)
```

peak learning rate、minimum learning rate 和 warmup steps 是预先选择的 [hyperparameters](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Model%20Hyperparameters.md>)；[Learning Rate Schedule](<Learning%20Rate%20Schedule.md>) 再根据当前 optimizer step 生成 $\alpha_t$。

> **Important**
> Cosine annealing 是 schedule，不是 line search。它不会先用一个 learning rate 找到 minimal loss 再换下一个，而是每次 parameter update 使用 schedule 给出的当前 $\alpha_t$。
