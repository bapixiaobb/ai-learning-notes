#AI #LanguageModeling

## What is perplexity?

就是 [next token](<Next-token%20prediction.md>) [cross-entropy loss](<Cross%20Entropy%20Loss.md>) 的一种表示方式：模型在每个位置的平均不确定程度

## Example

[LMs](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>) 在每个位置预测下一个 token

假设正确 token 是 `cat`，模型给它的概率是：
$ p(\text{cat}\mid \text{前文})=0.1. $
这个 token 对应的 negative log-likelihood 是：
$ -\log 0.1\approx2.30. $
把所有 token 的 negative log-likelihood 求平均，就得到平均 cross-entropy loss：
$ L = -\frac1T\sum_{t=1}^T \log p_\theta(x_t\mid x_{<t}). $
Perplexity 的定义只是：
$ \operatorname{PPL}=\exp(L). $

如果 $L=2.30$：
```math
\operatorname{PPL}=e^{2.30}\approx10.
```

>**Note**
> PPL $=10$ 可以粗略理解为：模型的平均不确定程度类似于在 10 个同样可能的选项中选择。它不表示模型真的只有 10 个候选 token。

## How to read it?

Perplexity 与 cross-entropy loss 包含相同信息：
```math
\operatorname{PPL}=e^L.
```

- loss 越低 $\Leftrightarrow$ perplexity 越低 $\Leftrightarrow$ next-token prediction 越好；
- 如果图中使用 **Negative Log-Perplexity**，因为 $-\log(\operatorname{PPL})=-L$，所以数值越高越好。

## [Scaling Law](<../02%20-%20Training%20and%20Scaling/Scaling%20Law.md>) and Limitation

Perplexity 是 pretraining 的 **upstream metric**。它通常会随 model parameters、data 或 compute 呈现比较平滑、可预测的 [scaling trend](<../02%20-%20Training%20and%20Scaling/Scaling%20Law.md>)。

但较低的 perplexity 不保证 downstream task（post- training 例如 classification、reasoning 或 SuperGLUE）一定更好：next-token prediction 的平均改善，不一定会直接转化成特定任务能力。因此 perplexity 是重要信号，但不能作为最终模型能力的唯一证据。
