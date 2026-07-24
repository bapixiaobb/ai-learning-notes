#AI #LanguageModeling

这是 2022 年 DeepMind 的 **Training Compute-Optimal Large Language Models**。Chinchilla 是论文里训练的模型名。这篇论文发现了：

****当 [training compute](<Training%20Compute%20-%206ND.md>) 固定时，当时很多 [大模型](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>) 都做得太大、训练数据却太少，也就是 undertrained。****

## Before Chinchilla

Kaplan scaling law 给出的趋势大致是：

```math
N_{\text{opt}}\propto C^{0.73}, \qquad D_{\text{opt}}\propto C^{0.27}
```

所以 compute 增加时，大部分新增 compute 被用来增加 parameters，data 增加得比较慢。

这推动了当时“把模型做得越来越大”的路线。

---
## Chinchilla 的发现

Chinchilla 重新做了大量 model-size 和 data-size 组合实验，发现 compute-optimal 的趋势更接近：

```math
N_{\text{opt}}\propto C^{0.5}, \qquad D_{\text{opt}}\propto C^{0.5}
```

也就是：**compute 增加时，model size 和 training tokens 应该大致同步增加。**

这经常被总结为 Chinchilla 的经验规则：

```math
D\approx20N
```

即每个 parameter 大约对应 20 个 training tokens。但这是他们实验设置中的 rule of thumb，不是永远正确的宇宙常数。

---
## 最有说服力的实验


![chinchilla.png](<../../attachments/chinchilla.png>)

DeepMind 原本有一个：

- Gopher：280B parameters，约 300B tokens。

后来使用几乎相同的 training compute，训练了：

- Chinchilla：70B parameters，约 1.3–1.4T tokens。

Chinchilla 的参数只有 Gopher 的四分之一，但 data 大约是四倍，最终在很多任务上反而表现更好。[Google DeepMind 的实验总结](https://deepmind.google/blog/an-empirical-analysis-of-compute-optimal-large-language-model-training/)

所以真正震撼的地方是：

```math
\boxed{ \text{同样的 training compute，模型更小但训练更久，反而可以更好} }
```

---

>**Important** — 如果考虑 inference/serving 成本，我们可能会故意选择比 Chinchilla-optimal 更小的模型，再用更多 tokens 训练它。因为小模型部署起来便宜，所以“training-compute optimal”不一定等于“整个产品生命周期 optimal”。
