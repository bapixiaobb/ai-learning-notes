#MachineLearning #DeepLearning #Optimization #LanguageModeling

>**Note**
> **Cross Entropy Loss** 是分类任务和 language modeling 中常用的 loss function，用来衡量模型预测的概率分布和真实标签分布之间的差异。

## 直觉

>**Note**
> Cross entropy 惩罚的是：模型有没有把足够高的概率分给正确答案。
>
> 如果正确类别的预测概率很高，loss 就小；如果正确类别的预测概率很低，loss 就大。

例如正确答案是 `cat`：

| 模型预测 | 对 `cat` 的概率 | Loss |
|---|---:|---:|
| 很确定是 cat | 0.95 | 小 |
| 不太确定是 cat | 0.40 | 中等 |
| 认为不是 cat | 0.01 | 很大 |

## 多分类形式

假设一共有 $K$ 个类别，真实标签是 one-hot vector：

```math
y = (y_1, y_2, \dots, y_K)
```

模型预测概率是：

```math
\hat{p} = (\hat{p}_1, \hat{p}_2, \dots, \hat{p}_K)
```

Cross entropy loss 定义为：

```math
\mathcal{L}
=
-
\sum_{k=1}^{K}
y_k \log \hat{p}_k
```

## Implementation
![TransformerLM.jpeg](<../../Transformer/TransformerLM.jpeg>)
一个完整的 [Transformer](<../../Transformer/Transformer.md>) 最后的输出是：
```
logits: [batch, seq, vocab_size]
```
cross-entropy loss 计算的是，对于在
```
logit: [B, S, :] = [...]
```
这有一个`vocab_size` 大小的 vector，是模型根据当前位置及之前的 tokens，对 vocabulary 中所有 next-token candidates 给出的未归一化分数。这里得到的只是分数，所以我们要做个 [Softmax](<../../Transformer/Softmax.md>)
```
probabilities = Softmax(logits): [batch, seq, vocab_size]
```
训练数据会给一个 `target: [batch, seq]` 这是一个索引，可以当成正确答案，意思是这个 `B, S, :] = [...]` 位置对应的 token 应该是 `[...]` 这个向量里的哪一个位置
```
correct_probabilities = gather(probabilities, target): [batch, seq]
```
取出正确 token 的 probability 后，计算 negative log。
```
losses = -log(correct_probabilities): [batch, seq]
```
然后再取个均值，得到一个 scalar
```
mean(losses)
```

>**Example** — `batch_size = 2, seq_len = 3, vocab_size = 5 -> probabilities.shape = [2,3,5]`
>假设 batch 0 的三个位置分别输出：
>```
>位置 0: [0.10, 0.60, 0.10, 0.10, 0.10]
>位置 1: [0.05, 0.10, 0.70, 0.10, 0.05]
>位置 2: [0.10, 0.10, 0.10, 0.20, 0.50]
>```
>batch 1：
>```
>位置 0: [0.80, 0.05, 0.05, 0.05, 0.05]
>位置 1: [0.10, 0.20, 0.10, 0.50, 0.10]
>位置 2: [0.05, 0.05, 0.10, 0.10, 0.70]
>```
>训练数据给出的正确 token ID：
>```
>targets = [
> [1, 2, 4],
> [0, 3, 4],
>]
>```
>它表示：
>```
>batch 0，位置 0，正确 token ID = 1
>batch 0，位置 1，正确 token ID = 2
>batch 0，位置 2，正确 token ID = 4
>
>batch 1，位置 0，正确 token ID = 0
>batch 1，位置 1，正确 token ID = 3
>batch 1，位置 2，正确 token ID = 4
>```
> 取出正确概率，逐个位置取：
>```
> probabilities[0,0,1] = 0.60
probabilities[0,1,2] = 0.70
probabilities[0,2,4] = 0.50
>
probabilities[1,0,0] = 0.80
probabilities[1,1,3] = 0.50
probabilities[1,2,4] = 0.70
>```
>得到：
>
>```
>correct_probabilities = [
>   [0.60, 0.70, 0.50],
>   [0.80, 0.50, 0.70],
>]
>```
## One-hot Label 情况

>**Note**
> 如果真实标签是 one-hot，那么只有正确类别那一项是 1，其余都是 0。

假设正确类别是 $c$，那么：

```math
y_c = 1
```

其余：

```math
y_k = 0, \quad k \neq c
```

因此 cross entropy 会化简成：

```math
\mathcal{L}
=
-\log \hat{p}_c
```

也就是：

>**Note**
> Cross entropy loss 只看模型给正确类别分配了多少概率。

## 与 Softmax 的关系

>**Note**
> Neural network 通常先输出 logits，而不是直接输出概率。Softmax 会把 logits 转换成概率分布。

给定 logits：

```math
z = (z_1, z_2, \dots, z_K)
```

[Softmax](<../../Transformer/Softmax.md>) 定义为：

```math
\hat{p}_k
=
\frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
```

然后 cross entropy 使用正确类别概率计算 loss：

```math
\mathcal{L}
=
-\log \hat{p}_c
```

## 在 Language Modeling 中

>**Note**
> 在 [Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 中，每一个位置都可以看作一个多分类问题：给定前面的 tokens，预测 vocabulary 中的下一个 token。

模型目标是：

```math
p_\theta(x_t \mid x_{<t})
```

如果正确的下一个 token 是 $x_t$，那么该位置的 loss 是：

```math
\mathcal{L}_t
=
-\log p_\theta(x_t \mid x_{<t})
```

整段序列的 loss 通常是所有 token positions 的平均：

```math
\mathcal{L}
=
-\frac{1}{T}
\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t})
```

## 与 Maximum Likelihood Estimation 的关系

>**Note**
> 最小化 cross entropy loss 等价于最大化模型给训练数据分配的 likelihood。

对于训练序列：

```math
x_1, x_2, \dots, x_T
```

language model 的 likelihood 是：

```math
p_\theta(x_1, \dots, x_T)
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
```

最大化 likelihood 等价于最小化 negative log-likelihood：

```math
-\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t})
```

这正是 cross entropy loss 的形式。

## Why Cross Entropy?

>**Note**
> Cross entropy 适合概率预测问题，因为它直接惩罚模型对正确答案分配的概率不足。

它常用于：

- classification
- [Next-token prediction](<./Next-token%20prediction.md>)
- [Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>)
- sequence modeling

## Related

- Loss Function
- [Softmax](<../../Transformer/Softmax.md>)
- [Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>)
- Maximum Likelihood Estimation
- Classification
- [Neural Network](<../../Neural%20Networks/Neural%20Network.md>)
- An Optimization Problem