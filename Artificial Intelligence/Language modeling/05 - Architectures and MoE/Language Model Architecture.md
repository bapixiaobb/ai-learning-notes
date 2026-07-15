#LanguageModeling #Architecture #Transformer

语言模型的 architecture 指 language model 内部的结构设计：token 进入模型后，经过哪些模块、以什么 shape 流动、如何进行信息交互、如何产生下一个 token 的 logits。

## 🧠 Core Idea

>**Note**
>Language Model Architecture 关注 language model 的结构设计。
>
>它的核心问题是：
>
>- language model 的结构由哪些 design choices 组成；
>- architecture 和 training recipe、systems 有什么区别；
>- Transformer 这一族模型如何演化到 modern LLM；
>- 为什么现代 LLM 多数采用 [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>)；
>- modern decoder-only LM 中有哪些常见 architecture pattern。

Lecture 3 的主线可以概括为：

>**Summary**
>**现代 LLM 的 architecture choices 有哪些？哪些已经形成共识？哪些只是 variation？**

其中最重要的入口概念是 [Model Architecture](<./Model%20Architecture.md>)。

## 🧭 Lecture 3 Main Thread (CS336)

当讨论一个 modern language model 的 architecture 时，不只是讨论：

- 模型有多少 layers；
- 模型有多少 parameters。

还包括：

- overall model family；
- attention variant；
- positional information；
- normalization；
- activation；
- MLP / FFN shape；
- residual path；
- output head。

这些内容共同构成 [Model Architecture](<./Model%20Architecture.md>)。

## 🧱 Architecture / Training / Systems

学习 language models 时，需要区分三条线：

| Concept | Main Question | Examples                                                                                   |
| ------------------------------- | ------------- | ------------------------------------------------------------------------------------------ |
| [Model Architecture](<./Model%20Architecture.md>) | 模型内部结构怎么设计？ | attention variant, norm, activation, positional embedding, MLP shape                       |
| [Training Recipe](<../02%20-%20Training%20and%20Scaling/Training%20Recipe.md>) | 模型如何被训练出来？ | [Optimizer](<../../Transformer/Optimizer.md>), [Learning Rate Schedule](<../../Transformer/Learning%20Rate%20Schedule.md>), batch size, weight decay, dropout, data mixture |
| [Systems for Language Models](<../03%20-%20GPU%20and%20Systems/Systems%20for%20Language%20Models.md>) | 模型如何被高效训练和推理？ | MFU, memory bandwidth, KV cache, parallelism, inference cost                               |

>**Note**
>这三条线相互影响，但不是同一个层面的概念。
>
>Architecture choices 会影响 systems cost；training recipe 会影响模型最终效果；systems constraints 也会反过来影响 architecture choices。
>
>但在理解 Lecture 3 时，主线仍然是 [Model Architecture](<./Model%20Architecture.md>)。

## 🧩 Basic Pipeline

一个 autoregressive language model 的基本结构可以粗略写成：

```math
\text{tokens}
\rightarrow
\text{token embeddings}
\rightarrow
\text{Transformer blocks}
\rightarrow
\text{final hidden states}
\rightarrow
\text{logits}
\rightarrow
p(x_{t+1} \mid x_{\leq t})
```

也就是：

1. [Tokenization](<../01%20-%20Language%20Modeling%20Basics/Tokenization.md>) 把 text 切成 token ids；
2. [Token Embedding](<../../Transformer/Token%20Embedding.md>) 把 token ids 映射成 continuous vectors；
3. [Positional Encoding](<../../Transformer/Positional%20Encoding.md>) / [Rotary Position Embedding](<../../Transformer/Rotary%20Position%20Embedding.md>) 注入顺序信息；
4. 多层 [Transformer Block](<../../Transformer/Transformer%20Block.md>) 更新 token representations；
5. Language Modeling Head 映射到 vocabulary logits；
6. [Softmax](<../../Transformer/Softmax.md>) 得到 next-token probability distribution。

## 🏗️ Architecture Foundations

### 1. Model Architecture

[Model Architecture](<./Model%20Architecture.md>) 解释 architecture 这个概念本身。

Architecture 不只是 number of layers，而是包括：

- depth / width；
- attention mechanism；
- position representation；
- normalization；
- activation；
- MLP shape；
- residual layout；
- output head。

>**Question**
>Architecture 到底包括哪些 design dimensions？
>
>它包括 model forward 的基本 [computation graph](<../../Neural%20Networks/Computational%20Graph.md>)、模块连接方式、hidden states 的 shape，以及每个主要模块的 design choices。

### 2. Training Recipe

[Training Recipe](<../02%20-%20Training%20and%20Scaling/Training%20Recipe.md>) 指模型结构确定后，训练这个模型所采用的一组方法和超参数。

它包括：

- [Optimizer](<../../Transformer/Optimizer.md>)；
- [Learning Rate Schedule](<../../Transformer/Learning%20Rate%20Schedule.md>)；
- batch size；
- weight decay；
- dropout；
- gradient clipping；
- data mixture；
- training tokens。

>**Note**
>两个模型可以有相同 architecture，但使用不同 training recipe。
>
>反过来，不同 architectures 也可以使用类似的 training recipe。

### 3. Systems for Language Models

[Systems for Language Models](<../03%20-%20GPU%20and%20Systems/Systems%20for%20Language%20Models.md>) 关注 language model 如何在硬件上高效训练和推理。

它包括：

- FLOPs；
- MFU；
- memory bandwidth；
- arithmetic intensity；
- KV cache；
- parallelism；
- inference cost；
- kernel optimization。

>**Note**
>Architecture choices 会直接影响 systems cost。
>
>例如 attention variant 会影响 KV Cache；MLP shape 会影响 FLOPs 和参数量；context length 会影响 attention memory。

### 4. Transformer Family

[Transformer Family](<../../Transformer/Transformer%20Family.md>) 描述 Transformer 相关模型之间的关系。

需要区分：

- [Transformer](<../../Transformer/Transformer.md>)
- Encoder-Only Transformer
- [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>)
- Encoder-Decoder Transformer
- [Llama-style Architecture](<../../Transformer/Llama-style%20Architecture.md>)

>**Note**
>现代 LLM 一般不是泛泛地“用了 Transformer”，而是更具体地使用了 [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>)。

### 5. Llama-style Architecture

[Llama-style Architecture](<../../Transformer/Llama-style%20Architecture.md>) 用来理解 modern decoder-only LM 的常见结构组合。

典型 choices 包括：

- [Pre-Norm Transformer](<../../Transformer/Pre-Norm%20Transformer.md>)
- [RMSNorm](<../../Transformer/RMSNorm.md>)
- [Rotary Position Embedding](<../../Transformer/Rotary%20Position%20Embedding.md>)
- [SwiGLU](<../../Transformer/SwiGLU.md>)
- [Causal Attention](<../../Transformer/Causal%20Attention.md>)
- Grouped Query Attention

>**Note**
>这里的 “Llama-style” 不是说所有 LLM 都是 Llama，而是用 Llama 代表一类 modern decoder-only Transformer 的 common design pattern。

## 🔬 Architecture Design Areas

| Area | Core Question | Related Concepts |
|---|---|---|
| block structure | 一个 Transformer block 内部怎么组织？ | [Transformer Block](<../../Transformer/Transformer%20Block.md>), [Pre-Norm Transformer](<../../Transformer/Pre-Norm%20Transformer.md>), [Residual Connection](<../../Transformer/Residual%20Connection.md>) |
| attention | token positions 如何交换信息？ | [Self-Attention](<../../Transformer/Self-Attention.md>), [Causal Attention](<../../Transformer/Causal%20Attention.md>), [Multi-Head Attention](<../../Transformer/Multi-Head%20Attention.md>), Grouped Query Attention |
| position | token 顺序如何表示？ | [Positional Encoding](<../../Transformer/Positional%20Encoding.md>), [Rotary Position Embedding](<../../Transformer/Rotary%20Position%20Embedding.md>), Relative Position Information |
| normalization | hidden states 如何稳定？ | [Layer Normalization](<../../Transformer/Layer%20Normalization.md>), [RMSNorm](<../../Transformer/RMSNorm.md>), [Pre-Norm Transformer](<../../Transformer/Pre-Norm%20Transformer.md>) |
| MLP | 每个 token representation 如何做 nonlinear processing？ | [MLP](<../../Transformer/MLP.md>), [Feed-Forward Network](<../../Neural%20Networks/Feed-Forward%20Network.md>), [SwiGLU](<../../Transformer/SwiGLU.md>), GELU |
| output | hidden states 如何变成 logits？ | Language Modeling Head, Logits, [Softmax](<../../Transformer/Softmax.md>) |

---

>**Summary** — My Understanding
>[Language Model Architecture](<./Language%20Model%20Architecture.md>) 这部分的核心不是记住某一个模型有多少层，而是建立一个关于 modern LLM 结构的整体认识：
>
>**architecture 是一组 design choices。**
>
>在 modern LLM 中，这些 choices 通常围绕 [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>) 展开，包括 attention、position embedding、normalization、MLP、activation、residual path 和 output head。
>
>Lecture 3 的学习重点是理解这些 choices 如何共同决定模型的表达能力、训练稳定性和计算成本。

## 🔗 Connections

- [Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>)
- [Large Language Model (LLM)](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>)
- [Model Architecture](<./Model%20Architecture.md>)
- [Training Recipe](<../02%20-%20Training%20and%20Scaling/Training%20Recipe.md>)
- [Systems for Language Models](<../03%20-%20GPU%20and%20Systems/Systems%20for%20Language%20Models.md>)
- [Transformer Family](<../../Transformer/Transformer%20Family.md>)
- [Original Transformer](<../../Transformer/Original%20Transformer.md>)
- [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>)
- [Llama-style Architecture](<../../Transformer/Llama-style%20Architecture.md>)
- [Transformer Block](<../../Transformer/Transformer%20Block.md>)
- [Self-Attention](<../../Transformer/Self-Attention.md>)
- [Causal Attention](<../../Transformer/Causal%20Attention.md>)
- [Multi-Head Attention](<../../Transformer/Multi-Head%20Attention.md>)
- Grouped Query Attention
- [Positional Encoding](<../../Transformer/Positional%20Encoding.md>)
- [Rotary Position Embedding](<../../Transformer/Rotary%20Position%20Embedding.md>)
- [Layer Normalization](<../../Transformer/Layer%20Normalization.md>)
- [RMSNorm](<../../Transformer/RMSNorm.md>)
- [SwiGLU](<../../Transformer/SwiGLU.md>)
- [MLP](<../../Transformer/MLP.md>)
- Language Modeling Head
- [Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)
- [Scaling Law](<../02%20-%20Training%20and%20Scaling/Scaling%20Law.md>)
