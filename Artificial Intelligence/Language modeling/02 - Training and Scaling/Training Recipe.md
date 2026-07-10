#LanguageModeling #Training #Optimization

[[Training Recipe]] 指训练一个 language model 时采用的一整套训练配置和训练策略。  
它不描述模型“长什么样”，而是描述：**在给定 [[Model Architecture]] 的情况下，如何把模型参数训练出来。**

## 🧠 Core Idea

>[!note]
>[[Model Architecture]] 决定模型内部 computation graph 的结构；[[Training Recipe]] 决定这个结构里的参数如何被学习。
>
>同一个 architecture 可以用不同 training recipes 训练，得到性能、稳定性和泛化能力完全不同的模型。

在 language modeling 中，一个 training recipe 通常包括：

- objective / loss function；
- [[Optimizer]]；
- [[Learning Rate Schedule]]；
- batch size；
- weight decay；
- dropout；
- gradient clipping；
- initialization；
- data mixture；
- number of training tokens；
- checkpointing / evaluation strategy。

这些 choices 不改变模型 forward pass 的基本结构，但会显著影响模型最终学到什么。

## 🧩 Architecture vs Training Recipe

一个 language model 可以分成两层理解：

| Concept | Question | Examples |
|---|---|---|
| [[Model Architecture]] | 模型内部结构怎么设计？ | attention variant, norm, activation, positional embedding, MLP shape |
| [[Training Recipe]] | 模型如何被训练出来？ | optimizer, learning rate, batch size, weight decay, dropout, data mixture |
| [[Systems for Language Models]] | 模型如何在硬件上高效训练和推理？ | MFU, memory bandwidth, parallelism, checkpointing, KV cache |

>[!important]
>Architecture 是模型结构；training recipe 是训练过程。
>
>例如 [[RMSNorm]]、[[Rotary Position Embedding]]、[[Grouped Query Attention]] 属于 architecture choices。
>
>但 learning rate、batch size、weight decay、optimizer 属于 [[Training Recipe]]。

## 🎯 Training Objective

Language model 的核心训练目标通常是 Next Token Prediction。

给定 token sequence：

```math
x_1, x_2, \dots, x_T
```

autoregressive language model 会学习：

```math
p_\theta(x_t \mid x_{<t})
```

整个 sequence 的 likelihood 可以写成：

```math
p_\theta(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
```

训练时通常最小化 negative log-likelihood：

```math
\mathcal{L}(\theta)
=
-
\sum_{t=1}^{T}
\log p_\theta(x_t \mid x_{<t})
```

也就是 cross-entropy loss。

>[!note]
>对 decoder-only language model 来说，training objective 和 architecture 是高度匹配的：
>
>[[Decoder-Only Transformer]] 用 Causal Mask 保证每个位置只能看见 prefix，然后通过 Language Modeling Head 输出 next-token logits。

## ⚙️ Optimizer

Optimizer 决定参数如何根据 gradient 更新。

最常见的是：

- [[SGD]]
- [[Adam]]
- [[AdamW]]

现代 LLM 训练中常用的是 [[AdamW]]，因为它把 weight decay 从 Adam 的梯度更新中 decouple 出来。

一个抽象的参数更新可以写成：

```math
\theta_{t+1}
=
\theta_t
-
\eta_t \cdot \mathrm{Update}(g_t)
```

其中：

- $\theta_t$ 是当前参数；
- $g_t$ 是 gradient；
- $\eta_t$ 是 learning rate；
- $\mathrm{Update}(g_t)$ 由 optimizer 决定。

>[!note]
>Optimizer 不改变模型结构。
>
>它改变的是参数在 loss landscape 中如何移动。

## 📉 Learning Rate Schedule

Learning rate schedule 决定训练过程中 step size 如何变化。

常见 choices 包括：

- constant learning rate；
- linear warmup；
- cosine decay；
- inverse square root decay；
- step decay。

在 LLM 训练中，常见模式是：

```math
\text{warmup}
\rightarrow
\text{decay}
```

warmup 阶段逐渐增大学习率，避免训练初期不稳定；decay 阶段逐渐降低学习率，让模型在训练后期更稳定地收敛。

>[!note]
>Learning rate 是 training recipe 中最重要的 hyperparameter 之一。
>
>它太大可能导致 training instability；太小则会导致训练效率低，甚至在有限 compute budget 下学不充分。

## 📦 Batch Size

Batch size 指一次参数更新使用多少 training examples / tokens。

在 language model training 中，更常见的是按 tokens 理解 batch size：

```math
\text{batch tokens}
=
\text{batch size}
\times
\text{sequence length}
```

如果使用 gradient accumulation，则 effective batch size 可以写成：

```math
\text{effective batch size}
=
\text{microbatch size}
\times
\text{gradient accumulation steps}
\times
\text{number of devices}
```

>[!note]
>Batch size 既是 training recipe 的一部分，也和 [[Systems for Language Models]] 强相关。
>
>较大的 batch size 通常能提高硬件利用率，但也会增加 memory pressure，并可能影响 optimization dynamics。

## 🧮 Gradient Accumulation

[[Gradient Accumulation]] 用来在显存不足以容纳大 batch 的情况下，模拟更大的 effective batch size。

>[!note]
>Gradient accumulation 不改变 model architecture，也不改变 objective。
>
>它改变的是一次 optimizer update 所对应的 effective batch size。

## 🧲 Weight Decay

Weight decay 是一种 regularization 方法，用来限制参数过大。

在普通形式下，weight decay 会鼓励参数范数较小：

```math
\mathcal{L}_{\text{total}}
=
\mathcal{L}
+
\lambda \|\theta\|_2^2
```

在 [[AdamW]] 中，weight decay 通常是 decoupled weight decay，也就是单独对参数做 shrinkage，而不是简单加到 gradient 里。

>[!note]
>Weight decay 属于 training recipe，不属于 architecture。
>
>它不改变模型结构，而是改变训练过程中对参数大小的偏好。

## 🌧️ Dropout

Dropout 是一种 regularization 方法。训练时随机把一部分 activations 置零，迫使模型不要过度依赖某些特定 hidden units。

在很多早期 Transformer 或较小模型中，dropout 很常见。  
但在大规模 LLM pretraining 中，dropout 的使用会因规模、数据量和训练设定而变化。

>[!note]
>Dropout 的位置和是否启用有时写在 architecture config 里，但它本质上更像 training recipe / regularization choice。
>
>因为 dropout 通常只在 training 时启用，在 inference 时关闭。

## ✂️ Gradient Clipping

Gradient clipping 用来避免 gradient norm 过大导致 training instability。

常见做法是 global norm clipping：

```math
g
\leftarrow
g
\cdot
\min
\left(
1,
\frac{c}{\|g\|}
\right)
```

其中 $c$ 是 clipping threshold。

>[!note]
>Gradient clipping 不改变 optimization objective。
>
>它限制的是每一步参数更新的最大幅度，常用于提升训练稳定性。

## 🧪 Initialization

Initialization 指训练开始前如何设置参数初值。

对于 deep Transformers，initialization 会影响：

- early training stability；
- gradient scale；
- activation scale；
- 是否容易出现 exploding / vanishing signals。

>[!note]
>Initialization 不是 architecture 本身，但它和 architecture 紧密相关。
>
>同样的 initialization 在 shallow model 和 very deep Transformer 中可能表现完全不同。

## 🗂️ Data Mixture

Data mixture 指训练数据中不同来源、不同类型、不同质量数据的比例。

例如，一个 language model 的 pretraining data mixture 可能包括：

- web text；
- books；
- code；
- academic papers；
- math data；
- multilingual data；
- instruction-style data。

>[!important]
>Data mixture 是 training recipe 中极其重要的一部分。
>
>对于 LLM 来说，模型能力不只来自 architecture 和参数量，也强烈依赖训练数据的组成和质量。

例如，一个模型是否擅长 code、math、reasoning、multilingual tasks，往往和 data mixture 高度相关。

## 🔢 Training Tokens

Training tokens 指模型在 pretraining 中看过多少 token。

在 scaling law 的语境中，模型性能通常同时受三件事影响：

- model size；
- dataset size / training tokens；
- compute budget。

>[!note]
>Training tokens 不属于 architecture。
>
>一个 7B model 可以训练 1T tokens，也可以训练 10T tokens。它们 architecture 相同，但 learned model 可能非常不同。

这也是为什么比较两个模型时，不能只看 parameter count，还要看 training tokens 和 data quality。

## 🧾 Checkpointing and Evaluation

Training recipe 还包括训练过程中如何保存和评估模型。

常见 choices：

- 每隔多少 steps 保存 checkpoint；
- 使用哪些 validation sets；
- 用 perplexity 还是 downstream benchmarks 评估；
- 是否保存 optimizer states；
- 是否做 intermediate evaluation。

>[!note]
>Checkpointing 在 systems 上也很重要，因为保存完整 model states 和 optimizer states 会带来显著 storage 和 I/O cost。
>
>但从 training recipe 角度看，它定义了训练过程如何被监控、恢复和选择最终模型。

## 🧠 Training Recipe as Optimization Setup

从 optimization 角度看，training recipe 定义了如何求解：

```math
\min_\theta \mathcal{L}(\theta)
```

其中：

- objective 决定优化什么；
- optimizer 决定怎么走；
- learning rate schedule 决定 step size 如何变化；
- batch size 决定 gradient estimate 的噪声；
- regularization 决定对参数或 activations 的偏好；
- data mixture 决定 loss 是在哪个数据分布上定义的。

>[!summary]
>Training recipe 不是一个单独参数，而是一整套 optimization setup。
>
>它决定模型如何从随机初始化走到一个具体的 trained model。

## ⚖️ Why It Matters

同一个 [[Model Architecture]]，如果使用不同的 [[Training Recipe]]，最终模型可能差异很大。

差异可能体现在：

- loss curve；
- convergence speed；
- training stability；
- final perplexity；
- downstream performance；
- reasoning ability；
- code ability；
- multilingual ability；
- robustness。

>[!note]
>Architecture 决定模型可以学什么；training recipe 很大程度上决定模型实际学到了什么。

---

>[!summary] My Understanding
>[[Training Recipe]] 描述 language model 的训练方案。
>
>它不是模型结构本身，而是在给定 [[Model Architecture]] 后，决定如何优化参数的一整套 choices。
>
>在 LLM 中，training recipe 至少包括 objective、optimizer、learning rate schedule、batch size、weight decay、dropout、gradient clipping、data mixture 和 training tokens。
>
>理解 training recipe 的关键是：
>
>**模型最终表现不只取决于 architecture，也取决于它如何被训练、用什么数据训练、训练了多少 tokens。**

## 🔗 Connections

- [[Language Model Architecture]]
- [[Model Architecture]]
- [[Systems for Language Models]]
- [[Cross Entropy Loss]] 
- [[Optimizer]]
- [[AdamW]]
- [[Learning Rate Schedule]]
- [[Batch Size]]
- [[Gradient Accumulation]]
- [[Weight Decay]]
- [[Dropout]]
- [[Data Mixture]]
- [[Training Tokens]]
- [[Scaling Law]]
- [[Resource Accounting]]