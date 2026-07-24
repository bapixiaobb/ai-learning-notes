#DeepLearning #NeuralNetwork #Optimization #LanguageModeling

## From Transformer LM to Loss
![TransformerLM.jpeg](<../attachments/TransformerLM.jpeg>)

这张图描述了 [Transformer](<../Transformer/Transformer.md>) LM 的 [forward](<Forward%20Propagation.md>)：

```math
\text{token IDs}
\rightarrow
\text{Transformer LM}
\rightarrow
\text{logits}
```

最后得到： `logits[b, s, :]` 都是一个长度为 `vocab_size` 的 vector，表示模型在位置 $s$ 对所有 next-token candidates 给出的 [未归一化分数](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Logits.md>)。

>**Important**
> Transformer LM 本身只负责根据 input tokens 产生 logits。它并不知道正确答案，所以 logits 还不是 loss。
>
> 在 training 中，loss function 位于 model forward 之后，同时接收 logits 和 targets，才能给模型这次 prediction 打分。

对于一段 token sequence，可以把相邻 tokens 错开一位：

```text
input:   x0, x1, ..., xS-1
target:  x1, x2, ..., xS
```

因此：

```text
Transformer LM(input) -> logits: [B, S, V]
next-token targets     -> target: [B, S]
```

## Language Modeling Uses Cross Entropy

[Next-token prediction](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Next-token%20prediction.md>) 在每个位置都是一个 `vocab_size` 类的 classification problem，所以 language model 通常使用 [Cross Entropy Loss](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)。

对位置 $(b,s)$：

1. `logits[b, s, :]` 给所有 vocabulary tokens 打分；
2. [Softmax](<../Transformer/Softmax.md>) 把这些 logits 解释成 probability distribution；
3. `target[b, s]` 指出正确的 next token；
4. cross entropy 取正确 token 的 probability，并计算：

```math
\mathcal L_{b,s}
=
-\log p_\theta\!\left(y_{b,s}\mid \text{prefix}_{b,s}\right)
```

再对 batch 和 sequence positions 求平均，得到一个 scalar loss：

```math
\mathcal L
=
-\frac{1}{BS}
\sum_{b=1}^{B}\sum_{s=1}^{S}
\log p_\theta\!\left(y_{b,s}\mid \text{prefix}_{b,s}\right)
```

概念上可以理解为 `Softmax + negative log`；实际实现通常直接从 logits 计算 numerically stable cross entropy。

## Loss Connects to Backpropagation and Optimizer

得到 scalar loss 之后，training 才能继续：

```math
\mathcal L
\xrightarrow{\text{backpropagation}}
\nabla_\theta \mathcal L
\xrightarrow{\text{optimizer}}
\text{updated parameters}
```

- Loss function 说明 model prediction 有多差；
- [Backpropagation](<Backpropagation.md>) 计算每个 parameter 对这个 loss 的影响，也就是 gradients；
- [Optimizer](<../Transformer/Optimizer.md>) 使用 gradients 更新 parameters，尝试让未来的 loss 变小。

>**Important**
> Optimizer minimize 的是 loss。它不负责产生 logits，也不负责定义或计算 loss。

## Loss Connects to Perplexity

对于使用 natural log 的平均 next-token cross-entropy loss：

```math
\operatorname{PPL}=\exp(\mathcal L)
```

因此 [Perplexity](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Perplexity.md>) 不是另一套 prediction，也不是 optimizer；它只是同一个 cross-entropy loss 的另一种表示方式：

```math
\mathcal L\downarrow
\quad\Longleftrightarrow\quad
\operatorname{PPL}\downarrow
```

训练时通常直接 minimize cross-entropy loss；评估和汇报 language model 时，可以把它转换成更容易解释的 perplexity。

## The Whole Training Chain

```math
\begin{aligned}
\text{input tokens}
&\rightarrow
\text{Transformer LM}
\rightarrow
\text{logits}\\
\text{logits}+\text{next-token targets}
&\rightarrow
\text{Cross Entropy Loss}
\rightarrow
\mathcal L\\
\mathcal L
&\rightarrow
\text{Backpropagation}
\rightarrow
\text{gradients}
\rightarrow
\text{Optimizer}
\rightarrow
\text{updated parameters}\\
\mathcal L
&\rightarrow
\exp(\mathcal L)
\rightarrow
\text{Perplexity}
\end{aligned}
```

>**Note**
> Inference 时没有正确的 next-token targets，因此通常不计算 loss。此时 logits 会经过 probability conversion 和 sampling / selection，用于 [Autoregressive Decoding](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>)。
