#LanguageModeling

**Engram 给 Transformer 增加了一条“根据短 token 组合查表”的路径，让模型用可训练的记忆向量辅助计算。** 它和 MoE 可以一起使用。[Engram Github](https://github.com/deepseek-ai/Engram)
## Why we need it?

语言里有很多反复出现的实体、短语和固定搭配。作者的出发点是：每次遇到这些组合，都让多层 Attention、FFN 重新构造其表示，可能不划算。把一部分常见模式存在表里，有望让主干网络把更多计算用于上下文处理和推理。[研究动机](https://arxiv.org/html/2601.07372v1#S1)

Engram 可以给当前 token 的向量，补充一份“附近几个 token 合在一起”的信息。**具体怎么走？看一个小例子。**

假设输入末尾是三个 token：`New / York / City`，当前处理 `City`。

1. 取以当前位置结尾的组合，如 `York City`、`New York City`，即 2-gram、3-gram。
2. 对组合做 hash，得到表的行号，读取 embedding。使用多个 hash head，缓解碰撞影响。
3. 拼接读出的向量，得到记忆表示 $e_t$。
4. 用当前 hidden state $h_t$ 决定这份记忆该用多少。

## Where is it?

加在选定 [Transformer Block](<../00%20-%20Maps%20and%20Architectures/Transformer%20Block.md>) 里的一个小模块，在该 block 的 Attention 之前，向 residual stream 添加一份信息。


![Engram Architecture](https://github.com/deepseek-ai/Engram/raw/main/figures/arch.png)

只有选定的几个 block 加了 Engram，不是每个 block 都加。

## System implementation of Engram
![Engram.png](<../../attachments/Engram.png>)
