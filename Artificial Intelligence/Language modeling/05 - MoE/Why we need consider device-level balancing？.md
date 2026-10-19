在 [MoE](<Mixture%20of%20Experts.md>) 的 imbalance 问题中，我们通常要分别考虑 expert imbalance 和 device imbalance 但是，天然的会有一个问题：

>**Question** — 如果每个 expert 都 balance 了，那每个 device 不就自然 balance 了吗？

>**Example** — 理论上，如果：
>>experts 被平均分到每个 device
>>每个 expert 收到完全一样多的 tokens
那会自然 device balance

但是实践中：**不能把 expert-level balancing loss 加得太强**

因为如果强行要求每个 expert 收到完全一样多的 tokens，会伤害 [MoE](<Mixture%20of%20Experts.md>) 的 specialization。

MoE 本来就希望不同 experts 学不同东西，如果太强迫均匀，[router](<MoE%20Router.md>) 就不能自由地把某类 token 多送给某些合适的 experts。

所以实践中常见三种不同层面的软约束：
- expert balancing 弱一些，避免 collapse
- device balancing 加一些，保证系统吞吐
- communication balancing 进一步避免 all-to-all traffic 某些路径太拥挤
