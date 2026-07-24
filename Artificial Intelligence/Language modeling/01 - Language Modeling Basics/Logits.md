#AI #LanguageModeling #DeepLearning

>**Note**
> **Logits** 是 model 最后一层输出的未归一化分数。它们不是 probability，也不是 loss。

在 decoder-only [Transformer](<../../Transformer/Transformer.md>) LM 中：

```math
\text{token IDs }[B,S]
\rightarrow
\text{logits }[B,S,V]
```

每个 `logits[b, s, :]` 都是长度为 vocabulary size $V$ 的 vector，表示模型在位置 $s$ 对所有 next-token candidates 的相对偏好。

同一份 logits 在两个阶段有不同去向：

- training：`logits + targets` $\rightarrow$ [Cross Entropy Loss](<Cross%20Entropy%20Loss.md>) $\rightarrow$ [loss](<../../Neural%20Networks/Loss%20Function.md>)；
- inference：最后一个位置的 logits $\rightarrow$ [Softmax](<../../Transformer/Softmax.md>) / decoding strategy $\rightarrow$ [next token](<Autoregressive%20Decoding.md>)。

>**Important**
> [Softmax](<../../Transformer/Softmax.md>) 把 logits 转成 probability distribution；cross entropy 再用正确 target 对这份 distribution 评分。
