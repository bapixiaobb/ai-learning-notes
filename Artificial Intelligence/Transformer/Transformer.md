#DeepLearning #NeuralNetwork #NLP #LanguageModeling #Architecture

[Transformer](<./Transformer.md>) 是一种处理 sequence data 的 Deep [Neural Network](<../Neural%20Networks/Neural%20Network.md>) Architecture。  
它的核心思想是：用 [Self-Attention](<./Self-Attention.md>) 让序列中每个 token 根据上下文更新自己的 representation。

## 🧠 Core Idea

>**Note**
>Transformer 的核心不是 recurrence，而是 attention-based sequence modeling。
>
>它不需要像 RNN 那样按时间顺序一个 token 一个 token 地递归处理序列，而是可以让所有 token positions 在同一层中并行地进行信息交互。

给定一个 token sequence：

```math
x_1, x_2, \dots, x_T
```

Transformer 会把每个 token 表示成 hidden vector，并通过多层 transformation 不断更新这些 hidden states：

```math
X^{(0)}
\rightarrow
X^{(1)}
\rightarrow
\cdots
\rightarrow
X^{(L)}
```

其中：

```math
X^{(\ell)} \in \mathbb{R}^{T \times d_{\text{model}}}
```

$T$ 是 sequence length，$d_{\text{model}}$ 是 hidden dimension。

![transformer model.png](<./transformer%20model.png>)
## 📍 Position in Deep Learning

>**Note**
>Transformer 是一种 [Neural Network](<../Neural%20Networks/Neural%20Network.md>) architecture，属于 [Deep Learning](<../Neural%20Networks/Deep%20Learning.md>)。
>
>在 NLP 中，现代 [Language Modeling](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 和 [Large Language Model (LLM)](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>) 大多基于 Transformer 或其变体。

可以理解成：

```math
\text{Transformer}
\subset
\text{Deep Neural Network Architecture}
```

在 language modeling 中，Transformer 通常作为一个参数化函数：

```math
f_\theta(\text{tokens}) \rightarrow \text{logits}
```

并用于建模：

```math
p_\theta(x_t \mid x_{<t})
```

但 Transformer 本身不是 language modeling objective。  
[Language Modeling](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 是任务目标，Transformer 是实现这个目标的一类 architecture。

## 🧩 Basic Structure

Transformer 通常由多个 [Transformer Block](<./Transformer%20Block.md>) 堆叠而成。

一个 Transformer block 通常包含：

- [Self-Attention](<./Self-Attention.md>)
- [MLP](<./MLP.md>)
- [Normalization](<./Normalization.md>)

粗略地看，一个 Transformer 会反复执行：

```math
\text{token representations}
\rightarrow
\text{contextualized token representations}
```

也就是说，每一层都会让 token representation 融合更多上下文信息。

>**Note**
>Transformer block 的具体排列方式会随着模型家族不同而变化。
>
>例如 [Original Transformer](<./Original%20Transformer.md>) 使用 encoder-decoder 结构，而现代 LLM 更多使用 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)。

## 🔁 Self-Attention as Token Mixing

[Self-Attention](<./Self-Attention.md>) 是 Transformer 的核心机制。

它让每个 token position 根据其他 token positions 的信息更新自己。

例如，在：

```text
river bank
bank account
```

这两个短语中，`bank` 的含义不同。  
Transformer 可以通过 self-attention 让 `bank` 关注周围 tokens，从而得到不同的 contextual representation。
>**Note**
>Self-attention 负责 token mixing。
>它决定一个 token representation 如何从其他 token positions 收集信息。

Self-attention 的具体公式和 $Q,K,V$ 机制放在 [Self-Attention](<./Self-Attention.md>) 中单独整理。

## **🧱 Transformer Block**

Transformer 的核心重复单元是 [Transformer Block](<./Transformer%20Block.md>)。

在现代 language model 中，一个 block 通常可以粗略理解为：

```math
\text{Attention}  
+  
\text{MLP}  
+  
\text{Residual Connection}  
+  
\text{Normalization}  
```

其中：

- [Self-Attention](<./Self-Attention.md>) 负责 token positions 之间的信息交互；
- [MLP](<./MLP.md>) 负责对每个 token representation 做 nonlinear transformation；
- [Residual Connection](<./Residual%20Connection.md>) 让信息沿着主路径稳定传递；
- [Normalization](<./Normalization.md>) 稳定 hidden states 的尺度。

> **Note**
> 不要把 Transformer block 的所有细节都放在 [Transformer](<./Transformer.md>) 里。

[Transformer](<./Transformer.md>) 只需要说明 block 是 Transformer 的基本堆叠单元；block 内部结构应该放在 [Transformer Block](<./Transformer%20Block.md>)。

## **🌳 [Transformer Family](<./Transformer%20Family.md>)**

Transformer 不是单一模型，而是一族 architecture。

从 [Original Transformer](<./Original%20Transformer.md>) 出发，可以发展出几类常见结构：

| **Architecture**                | **Typical Use**                                            |
| ------------------------------- | ---------------------------------------------------------- |
| Encoder-Only Transformer    | representation learning, classification, BERT-style models |
| [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)    | autoregressive language modeling, GPT/Llama-style models   |
| Encoder-Decoder Transformer | sequence-to-sequence tasks, translation, T5-style models   |
>**Note**
现代 LLM 通常不是泛泛地“用了 Transformer”，而是更具体地使用了 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)。
而 [Llama-style Architecture](<./Llama-style%20Architecture.md>) 可以看作 modern decoder-only Transformer 的一种代表性结构组合。

## **🧭 Transformer and Language Modeling**

Transformer 本身是一种 architecture；language modeling 是任务目标。

GPT-style language model 可以粗略理解为：

```math
\text{Decoder-Only Transformer}  
+  
\text{autoregressive language modeling objective}  
```

也就是建模：

```math
p_\theta(x_t \mid x_{<t})  
```
在 autoregressive language model 中，Transformer 位于 token embedding 和 output logits 之间：
```math
\text{raw text}  
\rightarrow  
\text{Tokenization}  
\rightarrow  
\text{token IDs}  
\rightarrow  
\text{Token Embedding}  
\rightarrow  
\text{Transformer}  
\rightarrow  
\text{logits}  
\rightarrow  
p_\theta(x_t \mid x_{<t})  
```
其中 [Causal Mask](<./Causal%20Mask.md>) 保证模型在预测当前位置时不能看到 future tokens。
## **⚖️ Transformer vs RNN**

| **Model**          | **Sequence Processing**  | **Strength** | **Limitation**                        |
| ------------------ | ------------------------ | ------------ | ------------------------------------- |
| RNN / LSTM | 按时间顺序递归处理                | 适合早期序列建模     | 难以并行，长距离依赖较难                          |
| [Transformer](<./Transformer.md>)    | 使用 self-attention 并行处理序列 | 易并行，适合大规模训练  | attention cost 随 sequence length 增长较快 |
>**Note**
Transformer 取代 RNN 的重要原因之一是它更适合 [GPU](<../Language%20modeling/03%20-%20GPU%20and%20Systems/GPU.md>) / TPU 上的大规模并行计算。
这也是为什么 Transformer 和 Systems for Language Models、[Resource Accounting](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 有很强的联系。

## **🔢 Numerical Linear Algebra View**

Transformer 的主要计算来自 matrix multiplication。

典型计算包括：

```math
XW_Q,\quad XW_K,\quad XW_V  
```

```math
QK^\top  
```

```math
\operatorname{Attention}(Q,K,V)  
```

```math
XW_1,\quad \sigma(XW_1)W_2  
```

>**Note**
从数值线性代数角度看，Transformer 的训练成本、FLOPs、memory bandwidth 和 hardware utilization 都与 Matrix Multiplication 密切相关。

更详细的 compute / memory 分析应该放在 [Resource Accounting](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>) 和 Systems for Language Models。

## **⭐ Why Transformer Matters**

Transformer 是现代 foundation models 和 large language models 的核心 architecture。

GPT、BERT、T5、LLaMA 等模型都基于 Transformer 或其变体。

它的重要性来自：

- 可以有效处理 sequence data；
- self-attention 能建模 long-range dependency；
- 比 RNN 更适合并行计算；
- 可以随着数据、参数和 compute scale up；
- 成为现代 [Language Modeling](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Language%20Modeling.md>) 的主流结构。

---

> **Summary** — My Understanding
[Transformer](<./Transformer.md>) 是一种 attention-based sequence model architecture。
>
>它的核心是用 [Self-Attention](<./Self-Attention.md>) 在 token positions 之间进行信息交互，再通过多层 [Transformer Block](<./Transformer%20Block.md>) 不断更新 token representations。
>
>在 language modeling 中，Transformer 通常作为从 token embeddings 到 logits 的主体网络。现代 LLM 多数使用 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)，而不是完整的 original encoder-decoder Transformer。

![TransformerLM.jpeg](<./TransformerLM.jpeg>)

