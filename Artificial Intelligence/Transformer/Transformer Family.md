#DeepLearning #NeuralNetwork #NLP #LanguageModeling #Architecture #Transformer

[[Transformer Family]] 指从 [[Original Transformer]] 发展出来的一组 Transformer-based architectures。  
它们共享 [[Self-Attention]]、[[Transformer Block]]、[[Residual Connection]]、[[Normalization]] 等核心思想，但在整体结构、attention mask、输入输出形式和训练目标上不同。

## 🧠 Core Idea

>[!note]
>Transformer 不是单一模型，而是一族 architecture。
>
>同样都叫 Transformer，可能指的是：
>
>- encoder-only model；
>- decoder-only model；
>- encoder-decoder model；
>- modern LLM 中的 Llama-style decoder-only model。
>
>这些模型都基于 attention-based representation learning，但它们解决的问题、可见上下文、输入输出形式不同。

在 [[Language Modeling]] 语境里，最需要区分的是：

```math
\text{Original Transformer}
\rightarrow
\text{Encoder-Decoder Transformer}
```

```math
\text{BERT-style}
\rightarrow
\text{Encoder-Only Transformer}
```

```math
\text{GPT/Llama-style}
\rightarrow
\text{Decoder-Only Transformer}
```

## 🧭 Why This Distinction Matters

>[!important]
>“Transformer” 这个词很容易造成混淆。
>
>有时候它指 Vaswani et al. 2017 的 encoder-decoder architecture；有时候它泛指 attention-based neural architecture；有时候它又特指 GPT/Llama 这类 decoder-only language model。

因此，在看 language model architecture 时，需要明确：

- 这个模型有没有 encoder；
- 这个模型有没有 decoder；
- attention 是 bidirectional 还是 causal；
- 训练目标是 masked token prediction、next-token prediction，还是 sequence-to-sequence prediction；
- 输出是每个 token 的 representation，还是下一个 token 的 logits。

## 🌳 Main Branches

| Branch | Typical Models | Main Use |
|---|---|---|
| [[Original Transformer]] / [[Encoder-Decoder Transformer]] | Transformer, T5, BART | translation, sequence-to-sequence |
| [[Encoder-Only Transformer]] | BERT, RoBERTa | representation learning, classification |
| [[Decoder-Only Transformer]] | GPT, LLaMA, Mistral | autoregressive language modeling |
| [[Llama-style Architecture]] | LLaMA-style open LLMs | modern decoder-only LLM pretraining / inference |

>[!note]
>[[Original Transformer]] 本身是 encoder-decoder structure。
>
>现代 LLM 中最常见的是 [[Decoder-Only Transformer]]，不是完整的 original Transformer。

## 🏛️ Original Transformer

[[Original Transformer]] 指 Vaswani et al. 2017 提出的 encoder-decoder Transformer。

>[!note]
>Original Transformer 的 decoder 里有两种 attention：
>
>- masked self-attention：看 target prefix；
>- cross-attention：看 encoder outputs。
>
>这和 modern decoder-only LLM 不同。GPT/Llama-style model 没有 encoder，也没有 cross-attention。

## 🧱 Encoder-Only Transformer

[[Encoder-Only Transformer]] 只保留 Transformer encoder。

它通常使用 bidirectional self-attention，也就是每个 token 可以看见序列中所有位置。

```math
h_t
\text{ can attend to }
h_1, h_2, \dots, h_T
```

这类模型适合学习 contextual representations。

典型模型：

- BERT；
- RoBERTa；
- DeBERTa。

常见任务：

- classification；
- token classification；
- sentence embedding；
- retrieval；
- masked language modeling。

>[!note]
>Encoder-only Transformer 适合理解输入，但不天然适合 autoregressive generation。
>
>因为 bidirectional attention 会让模型看到未来 tokens，这和 next-token prediction 的因果约束不匹配。

## 🧮 Decoder-Only Transformer

[[Decoder-Only Transformer]] 只保留 Transformer decoder 中的 causal self-attention 部分。

它没有 encoder，也通常没有 cross-attention。

每个 token position 只能看到 prefix：

```math
h_t
\text{ can attend to }
h_{\leq t}
```

这类模型天然适合 [[Autoregressive Language Model]]：

```math
p_\theta(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^T
p_\theta(x_t \mid x_{<t})
```

典型模型：

- GPT；
- LLaMA；
- Mistral；
- Qwen；
- DeepSeek-style decoder-only LMs。

>[!note]
>现代 LLM 多数是 decoder-only Transformer。
>
>它们的核心训练目标通常是 [[Next Token Prediction]]，并通过 [[Causal Mask]] 保证模型不能看到 future tokens。

## 🔁 Encoder-Decoder Transformer

[[Encoder-Decoder Transformer]] 同时包含 encoder 和 decoder。

它适合 sequence-to-sequence tasks，例如：

- machine translation；
- summarization；
- text-to-text generation；
- conditional generation。

结构上可以理解为：

```math
\text{input sequence}
\rightarrow
\text{encoder representations}
\rightarrow
\text{decoder}
\rightarrow
\text{output sequence}
```

decoder 在生成时依赖两类信息：

1. 已经生成的 target prefix；
2. encoder 对 source sequence 的 representations。

>[!note]
>Encoder-decoder Transformer 适合“给定一个输入序列，生成另一个输出序列”的任务。
>
>而 decoder-only Transformer 更适合把所有内容统一成一个 autoregressive token stream。

## 🦙 Llama-style Architecture

[[Llama-style Architecture]] 是 modern decoder-only LLM 的一种代表性 architecture pattern。

它通常不是简单复刻 [[Original Transformer]]，而是在 decoder-only Transformer 基础上采用一组 modern design choices：

| Component | Common Choice |
|---|---|
| overall structure | [[Decoder-Only Transformer]] |
| attention | [[Causal Attention]] |
| position information | [[Rotary Position Embedding]] |
| normalization | [[RMSNorm]] |
| block layout | [[Pre-Norm Transformer]] |
| MLP activation | [[SwiGLU]] |
| attention variant | [[Grouped Query Attention]] in many modern models |

>[!note]
>“Llama-style” 不是说所有 modern LLM 都是 LLaMA，而是指一类常见的 modern decoder-only Transformer design pattern。
>
>这也是 CS336 lecture 3 里讨论 modern LLM architecture choices 时很重要的一条线。

## 🔍 Attention Pattern Comparison

不同 Transformer branches 的一个关键区别是 attention pattern。

| Architecture | Attention Pattern | Can See Future Tokens? | Typical Objective |
|---|---|---|---|
| [[Encoder-Only Transformer]] | bidirectional self-attention | Yes | masked language modeling / representation learning |
| [[Decoder-Only Transformer]] | causal self-attention | No | next-token prediction |
| [[Encoder-Decoder Transformer]] | encoder bidirectional + decoder causal + cross-attention | decoder cannot see future target tokens | sequence-to-sequence prediction |

>[!important]
>Attention mask 决定一个 token position 能看见哪些 positions。
>
>这会直接决定模型适合做“理解”、”生成“，还是 conditional generation。

## 🎯 Objective Comparison

Transformer family 的不同分支通常和不同 training objective 绑定。

| Architecture | Common Objective | Output Meaning |
|---|---|---|
| [[Encoder-Only Transformer]] | masked language modeling | 学习 contextual representation |
| [[Decoder-Only Transformer]] | next-token prediction | 预测下一个 token |
| [[Encoder-Decoder Transformer]] | conditional generation | 根据 source sequence 生成 target sequence |

>[!note]
>Architecture 和 objective 不是完全绑定死的，但它们通常互相匹配。
>
>例如 [[Decoder-Only Transformer]] 和 [[Next Token Prediction]] 非常匹配，因为 causal self-attention 刚好对应 autoregressive factorization。

## 🧩 Relation to Language Modeling

在 modern [[Language Modeling]] 中，最重要的分支是：

```math
\text{Transformer}
\rightarrow
\text{Decoder-Only Transformer}
\rightarrow
\text{Llama-style Architecture}
```

这是因为 autoregressive language modeling 需要建模：

```math
p_\theta(x_t \mid x_{<t})
```

而 decoder-only Transformer 的 causal attention 正好保证：

```math
x_t
\text{ cannot use information from }
x_{>t}
```

>[!note]
>因此，当讨论 modern LLM architecture 时，“Transformer” 通常更具体地指 decoder-only Transformer，而不是 original encoder-decoder Transformer。

---

>[!summary] My Understanding
>[[Transformer Family]] 这张卡的核心是区分不同 Transformer-based architectures。
>
>[[Original Transformer]] 是 encoder-decoder structure；[[Encoder-Only Transformer]] 适合 representation learning；[[Decoder-Only Transformer]] 适合 autoregressive language modeling；[[Llama-style Architecture]] 则是 modern decoder-only LLM 的一种常见结构组合。
>
>理解这组关系后，看到“Transformer”这个词时，就需要判断它到底是在指 original Transformer、Transformer family，还是 modern decoder-only LLM。

## 🔗 Connections

- [[Transformer]]
- [[Language Model Architecture]]
- [[Model Architecture]]
- [[Original Transformer]]
- [[Encoder-Only Transformer]]
- [[Decoder-Only Transformer]]
- [[Encoder-Decoder Transformer]]
- [[Llama-style Architecture]]
- [[Self-Attention]]
- [[Causal Attention]]
- [[Causal Mask]]
- [[Cross-Attention]]
- [[Next Token Prediction]]
- [[Autoregressive Language Model]]
- [[Masked Language Modeling]]
- [[Language Modeling]]
- [[Large Language Model (LLM)]]