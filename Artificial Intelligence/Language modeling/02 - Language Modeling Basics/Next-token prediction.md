#AI #LanguageModeling #LLM

# 什么是 next-token prediction 的理论目标？

LLM 训练的核心目标是：**给定前面的 tokens，预测下一个 token。**

比如：
```
The capital of France is ___
```
模型应该给 `Paris` 高概率（这里假设 `Paris` 是一个 [token](<Tokenization.md>)）。

这里的输入：
```
x = The capital of France is
```

模型的输出：
```
f(x) = logits of the next token
```
logits：它们只是模型给每个 token 的原始分数，可以为负，也不要求加起来等于 1。
## Example

假设这是一个只有 5 个候选 token 的 toy vocabulary:
```
vocab  = [Beijing, NewYork, London, Paris, Berlin]
logits = [0,       0,       0,      2,     0]
```

经过 [Softmax](<../../Transformer/02%20-%20Attention/Softmax.md>) 把 logits 转成
```
probability ≈ [0.088, 0.088, 0.088, 0.649, 0.088]
```

训练时，我们最主要优化的是：**让真实下一个 token 的 probability 越高越好**

也就是常说的：minimize [Cross Entropy Loss](<Cross%20Entropy%20Loss.md>)，或者 maximize likelihood of next token

这就是 **next-token prediction objective**。
