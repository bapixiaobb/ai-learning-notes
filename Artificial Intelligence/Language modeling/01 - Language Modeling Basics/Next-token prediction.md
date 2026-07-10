#AI #LanguageModeling #LLM 

# 什么是 next-token prediction 的理论目标？

LLM 训练的核心目标是：**给定前面的 tokens，预测下一个 token。**

比如：
```
The capital of France is ___
```
模型应该给 `Paris` 高概率。

训练时，我们最主要优化的是：==**让真实下一个 token 的 probability 越高越好

也就是常说的：minimize [[Cross Entropy Loss]]，或者 maximize likelihood of next token

这就是 **next-token prediction objective**。

>[!note] 它回答的是：
>**模型预测下一个 token 预测得准不准？**


