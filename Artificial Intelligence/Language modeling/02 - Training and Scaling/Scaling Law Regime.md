#AI #LanguageModeling

[Scaling Law](<Scaling%20Law.md>) 的完整问题其实是：
$ \text{scale（参数、数据、compute）} \longrightarrow \text{某个 performance metric} $
如果拟合的是：
$ \text{model size} \longrightarrow \text{cross-entropy loss / perplexity} $
那它能支持的结论只是：

> 模型变大时，next-token prediction 的平均 [loss](<../../Neural%20Networks/Loss%20Function.md>)/[perplexity](<../01%20-%20Language%20Modeling%20Basics/Perplexity.md>) 会怎样变化。

它不能自动推出：

$ \text{model size} \longrightarrow \text{reasoning、classification、SuperGLUE accuracy} $

原因是 perplexity 是一种“平均分”。例如模型把常见词、标点和简单句预测得更准，也能降低平均 perplexity，但这不一定说明它的数学推理能力提高了。
```math
\boxed{ \text{PPL scaling 很稳定} \not\Rightarrow \text{downstream performance scaling 也稳定} }
```

- upstream (pre-training)：模型预测下一个 token 好不好
    → 用 cross-entropy / perplexity 衡量，通常变化平滑，容易拟合 scaling law。

- downstream (post-training)：模型能不能完成分类、推理、问答等具体任务
    → 可能受到任务分布、prompt、fine-tuning 和能力阈值等因素影响，变化不一定平滑。
