#AI #LanguageModeling

[MoE](<Mixture%20of%20Experts.md>) 中决定每个 token 应该送到哪些 experts。

Router 负责为每个 token 计算 experts scores，并选择 top-k experts。
对于 token representation $x$，router 可以产生 $$s=Router(x)$$
其中：$$s\in\mathbb{R}^N$$ 表示这个 token 对 $N$ 个 experts 的 scores。
然后选出 top-k experts: $$\mathrm{TopK}(s)$$
>**Note**
>在很多 MoE 里，routing 是 token-level 的。
>也就是每个 token 都会单独决定自己要去哪个 expert，而不是整句话统一送到同一个 expert。

## 🔀 Top-k Routing

Top-k routing 是 MoE 中最常见的 routing 方式之一。
流程可以理解为：
1. 对每个 token 计算 experts scores
2. 选出分数最高的 $k$ 个 experts
3. 只把 token 送到这些 experts
4. 把 experts outputs 按 gate weights 加权组合
形式上:
```math
y=\sum_{i\in\mathrm{TopK}(s)}g_iE_i(x)
```
其中：
- $E_i(x)$ 是第 $i$ 个 expert 的输出
- $g_i$ 是 router / gate 给这个 expert 的权重
- $\mathrm{TopK}(s)$ 是被选中的 experts

>**Note** — Top-k routing 的直觉：
>每个 token 只找最适合自己的少数几个 experts，而不是让所有 experts 都处理它

## Training through top-k routing

>**Tip** — 一旦用了 [top-k routing](<MoE%20Router.md>)，routing decision 就会变得很硬。每个 token 只看到自己被分配到的 experts，没被选中的 experts 是一种 counterfactual：
>
>>**Question** — **我们不知道”如果这个 token 给别的 expert 处理会怎么样“**

这就产生了两个训练问题：
1. routing 本身不是普通的光滑 differentiable path
2. 没被选中的 expert 得不到这个 token 通过 expert computation 产生的 gradient signal
	- 如果 expert 没被选中，它没有参与 [forward computation](<../../Neural%20Networks/Forward%20Propagation.md>)，那么这个 token 的 language modeling loss 就不会通过这个 expert 的 MLP [反传](<../../Neural%20Networks/Backpropagation.md>)。
	- 但 router 这边还可能通过 selected gates、[Softmax](<../../Transformer/02%20-%20Attention/Softmax.md>) probability、auxiliary loss 等收到某些 gradient。

所以 MoE 的 router 很容易变成一个不稳定的 **selection system**
