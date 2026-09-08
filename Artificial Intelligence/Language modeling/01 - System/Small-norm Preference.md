#AI #LanguageModeling

一个 [LLM](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20%28LLM%29.md>) 通常可以用很多组不同的 parameters 拟合训练数据：

```math
\theta^{(1)},\theta^{(2)},\ldots
```

它们的 [training loss](<../02%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>) 可能差不多，但有些解依赖非常大的 weights，甚至依赖多个大数之间的精细抵消。这种解可能更容易：

- 拟合训练数据中的偶然噪声；
- 对输入或参数的微小变化更敏感；
- 在训练集之外表现较差；
- 让 parameter magnitude 在训练中持续增长。

因此，我们人为加入一种 small-norm preference：

> 如果两组 parameters 都能很好地解释训练数据，那么稍微偏好 norm 较小的那一组。

>**Important** — 在没有足够数据证据时，不要让 parameter magnitude 无限制地增大。

这种偏好可以通过在 optimization 中加入 [Weight Decay](<Weight%20Decay.md>) 来实现。
