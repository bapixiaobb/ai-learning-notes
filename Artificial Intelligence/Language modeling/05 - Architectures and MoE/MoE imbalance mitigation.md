#AI #Transformer #LLM #NLP

## ❓ Why MoE 天然会有 imbalance 问题

**MoE 的目标是**：不要每个 token 都激活所有 experts。

如果每个 token 都跑所有 experts，那就退化成 dense model。所以 MoE 必须 sparse：每个 token 只选 top-k 个 experts。

>**Question** — 一旦用了 top-k routing，routing decision 就会变得很硬。每个 token 只看到自己被分配到的 experts，没被选中的 experts 是一种 counterfactual：
>
>**我们不知道”如果这个 token 给别的 expert 处理会怎么样“**

这就产生了两个训练问题：
1. routing 本身不是普通的光滑 differentiable path
2. 没被选中的 expert 得不到这个 token 通过 expert computation 产生的 gradient signal
	- 如果 expert 没被选中，它没有参与 forward computation，那么这个 token 的 language modeling loss 就不会通过这个 expert 的 MLP 反传。
	- 但 router 这边还可能通过 selected gates、[Softmax](<../../Transformer/Softmax.md>) probability、auxiliary loss 等收到某些 gradient。
所以 MoE 的 router 很容易变成一个不稳定的 **selection system**

---
## ❓ Rich-get-richer：why expert collapse

>**Note** — Intuition
>一开始 router 选中了某个 expert。 这个 expert 得到了更多的 token，也得到了更多 gradient。
>它训练得更好之后，router 更倾向于继续选择它。于是它越来越强、越来越常被选中。

这是一个正反馈：更常被选中 ➡️ 得到更多训练信号 ➡️ 变得更好 ➡️ 更常被选中

最后就会出现两个现象：
**Expert collapse**: 少数 experts 吞掉大部分 tokens。
**Expert starvation**: 某些 experts 几乎没有 token，也就学不到东西

这个会导致 MoE 名义上有很多 experts，但实际只有少数 experts 在工作。换句话说，你买了很多 total parameters，但很多参数根本没有被有效训练和使用。

---
## ❓ MoE 中的 stragger problem

Dense Transformer 中，每个 token 都经过同一套 layer，所以负载比较规则。

MoE 里，每个 token 的路径取决于 router：

```
token A -> expert 1, expert 7
token B -> expert 3, expert 12
```

如果这些 experts 分布在不同的 devices 上， activation 就要跨设备移动。
所以 MoE 训练里经常会出现 all-to-all communication：

> 每张 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 可能要把 token activations 发给其他的卡

这个 communication 如果不均衡，就会拖慢训练

---
## 📎 Expert balancing vs. Device balancing

Expert balancing 更像 optimization 问题：每个 expert 有没有足够训练信号？
Device balancing 更像 systems 问题：每台 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 的计算和通信负载是不是均衡？

### 为什么要单独考虑 device-level balancing？

>**Question** — 如果每个 expert 都 balance 了，那每个 device 不就自然 balance 了吗？
> 理论上，如果：
> > experts 被平均分到每个 device
> > 每个 expert 收到完全一样多的 tokens
> 那会自然 device balance

但是实践中：**你不能把 expert-level balancing loss 加得太强**

因为如果强行要求每个 expert 收到完全一样多的 tokens，会伤害 MoE 的 specialization。

MoE 本来就希望不同 experts 学不同东西，如果太强迫均匀，router 就不能自由地把某类 token 多送给某些合适的 experts。

所以实践中常见三种不同层面的软约束：
- expert balancing 弱一些，避免 collapse
- device balancing 加一些，保证系统吞吐
- communication balancing 进一步避免 all-to-all traffic 某些路径太拥挤

---
## ✅ Solution

### 1. RL routing
把 router 看成一个 policy，用 RL 或 reinforce 之类的方法学 routing decision。这个理论上可以做，但是实践不主流，因为 gradient variance 大，系统复杂，而且效果不一定比 heuristic 好。

### 2. Stochastic perturbation
给 router 加噪音

**Main idea:** 不要让 router 太早做完全正确的 hard decision，而是在 routing score 上加一点 stochastic noise。如果两个 experts 分数很接近，noise 可以帮助模型探索不同 experts，也可以打破 early tie。

这种方法的作用更像 exploration：让一些 experts 有机会被试到，而不是一开始就被 router 排除掉。但这类 stochastic robustness trick 不一定是必要的，有些后续的 ablation 发现去掉它反而能更稳定或者质量更好。

### 3. 🌟 Heuristic balancing losses (main stream)
实际训练 MoE 时，大家通常会在 normal language modeling loss 之外，加一个额外的 balancing loss。

总目标可以理解成：language modeling loss + 一个控制 expert 使用均匀性的 auxiliary loss。

这个 auxiliary loss 不是从 [Next-token prediction](<../01%20-%20Language%20Modeling%20Basics/Next-token%20prediction.md>) 的理论目标中自然推出来的，而是一个工程上非常有效的 heuristic: 它逼迫 router 不要把所有 tokens 都送到少数 experts。

>**Note** — 这个方法它听起来有点“不优雅”，但现实里很有用
>从纯理论角度看，language model 只应该优化 next-token prediction loss。
>但 MoE 实践中，如果你只优化 next-token prediction loss，router 很容易 collapse。
>所以工程上不得不加一个额外目标：别让 experts 用得太不均匀。

#### switch Transformer-style load balancing loss
经典的 load balancing loss 大概有两个量：
- 每个 expert 实际收到多少 tokens
- router 给每个 expert 分配多少 probability mass
直觉上，如果某个 expert 已经收到很多 tokens，那loss 就会惩罚 router 继续给它很高的概率。也就是说，它会把 popular experts 的 routing probability 往下压（就是防爆）。

所以这个 loss 的作用是：
- 如果某个 expert 太热门，loss 会对它的 routing probability 施加负反馈
- 冷门 experts 则通过 probability mass 的重新分配，间接获得更多被选中的机会。

这可以抵消 rich-get-richer 的问题
### 4. Per-expert bias

DeepSeek V3 使用的一种均衡方法，它的直觉是：给每个 expert 的 routing score 加一个可调 bias

- 如果某个 expert 被用的太多，就降低它的bias
- 如果某个 expert 被用的太少，就提高它的bias

这样 router 在选择 experts 时，就会受到这个 bias 影响，从而动态调节 expert 使用频率。

这个方法比直接加 auxiliary loss 更接近 online adjustment：它不是只在 loss 里惩罚 imbalance，而是直接修改 expert 被选中的倾向。

虽然 DeepSeek V3 这类方法试图减少那些“丑陋的 auxiliary losses”，但目前还没有完全摆脱它们。为了避免 extreme imbalance，很多时候仍然需要一些 auxiliary balancing 机制。

---
## Difference

- Load balancing loss 是加到 objective 里的额外惩罚。  
- Router noise 是在 routing score 上加扰动，增加 exploration。  
- Per-expert bias 是直接调整每个 expert 被选中的倾向。  
- Per-device balancing 是从系统利用率角度，让不同设备负载更均衡。
- 这些都属于 MoE imbalance mitigation heuristics
	- load balancing loss 和 per-device balancing loss 是 auxiliary objectives
	- router noise 和 per-expert bias 更像 routing-time / online adjustment tricks。


>**Summary** — My understanding
> MoE 训练里有 top-k selection、不连续 routing 和 counterfactual experts 这些难处理的问题。实践中通常不会显式求解完整的 routing / bandit 问题，而是只对被选中的 experts 正常 backprop，同时加入 load balancing loss 抑制 rich-get-richer 和 expert collapse。这个简单的 heuristic 组合在经验上居然能稳定训练 MoE。


# 🔗
[Mixture of Experts (MoE)](<./Mixture%20of%20Experts%20(MoE).md>)
[Transformer](<../../Transformer/Transformer.md>)