#LanguageModeling #Architecture #ModelDesign

[[Model Architecture]] 指模型内部 computation structure 的整体设计。  
它描述模型由哪些模块组成、这些模块如何连接、hidden states 以什么 shape 流动，以及每个主要模块采用什么 design choices。

## 🧠 Core Idea

>[!note]
>Architecture 不是只指“模型有几层”。
>
>在 language model 中，architecture 至少包括：
>
>- overall model family；
>- depth / width；
>- attention mechanism；
>- positional information；
>- normalization；
>- activation function；
>- MLP / FFN shape；
>- residual layout；
>- output head。
>
>这些共同决定模型内部的 computation graph。

更具体地说，architecture 决定了：

$$
f_\theta:
(x_1, x_2, \dots, x_t)
\mapsto
p_\theta(x_{t+1} \mid x_{\leq t})
$$

这个函数族的形式。

其中：

- architecture 决定 $f_\theta$ 的结构；
- parameters $\theta$ 决定这个结构里的具体数值；
- training recipe 决定如何学习这些参数。

## 🧱 Architecture is More Than Number of Layers

很多时候说一个模型“有多少层”，只是描述了 architecture 的 depth。

但两个模型即使有相同的 number of layers，也可能有完全不同的 architecture。

例如，两个 32-layer decoder-only Transformers 可能在以下方面不同：

| Design Dimension | Example Choices |
|---|---|
| attention variant | [[Artificial Intelligence/Transformer/Multi-Head Attention]], [[Multi-Query Attention]], [[Grouped Query Attention]] |
| positional information | [[Absolute Positional Embedding]], [[Rotary Position Embedding]], [[ALiBi]] |
| normalization | [[Layer Normalization]], [[RMSNorm]] |
| norm placement | [[Pre-Norm Transformer]], [[Post-Norm Transformer]] |
| MLP activation | [[GELU]], [[SwiGLU]] |
| MLP shape | FFN expansion ratio, gated MLP |
| residual path | sequential residual, parallel residual |
| output design | tied embedding, untied embedding |
| context handling | fixed context length, long-context extension |

>[!important]
>Number of layers 只说明模型有多深。
>
>Architecture 还包括模型每一层里面具体做什么、怎么连接、用什么 operation，以及 hidden representations 如何在这些 modules 之间流动。

## 🧩 Architecture as Computation Graph

从 computation graph 的角度看，architecture 决定模型的信息流动方式。

一个 language model 的 forward pass 可以粗略理解为：

$$
\text{tokens}
\rightarrow
\text{embeddings}
\rightarrow
\text{blocks}
\rightarrow
\text{hidden states}
\rightarrow
\text{logits}
$$

在这个过程中，architecture 决定：

- token ids 如何变成 vectors；
- 每一层如何更新 hidden states；
- token positions 如何交换信息；
- nonlinear transformation 在哪里发生；
- normalization 在哪里发生；
- residual path 如何保留信息；
- final hidden state 如何变成 vocabulary logits。

>[!note]
>Architecture 不是训练出来的结果，而是训练开始之前已经确定的 structure。
>
>Training 会改变参数值，但不会改变模型的 computation graph，除非使用了 dynamic architecture 或 architecture search。

## 🧬 Architecture as Function Class

从数学角度看，architecture 定义了模型可以表示的函数族。

一个 autoregressive language model 可以看成：

$$
p_\theta(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
$$

其中每一项：

$$
p_\theta(x_t \mid x_{<t})
$$

由 model architecture 和 parameters $\theta$ 共同决定。

>[!note]
>Architecture 决定 possible functions。
>
>Training 决定 learned function。

也可以理解为：

$$
\text{Architecture}
\Rightarrow
\text{hypothesis class}
$$

$$
\text{Training}
\Rightarrow
\text{specific model within that class}
$$

这也是为什么仅仅比较 parameter count 不够：  
两个参数量相近的模型，如果 architecture 不同，表达能力、训练稳定性、推理成本都可能不同。

## 🏗️ Main Design Dimensions

### 1. Overall Model Family

Model architecture 首先包含模型所属的整体结构族。

在 language modeling 中常见的 model families 包括：

- [[Encoder-Only Transformer]]
- [[Decoder-Only Transformer]]
- [[Encoder-Decoder Transformer]]
- [[Mixture of Experts (MoE)]]

>[!note]
>Modern LLM 最常见的是 [[Decoder-Only Transformer]]。
>
>它天然适合 autoregressive next-token prediction，因为每个位置只需要基于 prefix 预测下一个 token。

### 2. Depth and Width

Depth 和 width 是最直观的 architecture shape choices。

| Choice | Meaning |
|---|---|
| number of layers | 有多少个 Transformer blocks |
| hidden dimension | 每个 token representation 的维度 |
| number of heads | attention 分成多少个 heads |
| head dimension | 每个 attention head 的维度 |
| intermediate size | MLP hidden layer 的维度 |
| context length | 模型一次最多处理多长的 sequence |

>[!note]
>Depth 决定模型经过多少次 representation transformation。
>
>Width 决定每个 token representation 有多大的表达空间。

这些 choices 会影响：

- parameter count；
- FLOPs；
- activation memory；
- KV cache size；
- model capacity；
- training stability。
### 3. Attention Design

Attention design 决定 token positions 之间如何交换信息。

Architecture 中常见的 attention choices 包括：

- 是否使用 [[Self-Attention]]；
- 是否使用 [[Causal Attention]]；
- 使用 [[Artificial Intelligence/Transformer/Multi-Head Attention]]、[[Multi-Query Attention]] 还是 [[Grouped Query Attention]]；
- $Q,K,V$ 的 shape 如何设计；
- 是否使用 sliding window attention；
- context length 如何影响 attention computation 和 [[KV Cache]]。

>[!note]
>Attention 是 token mixing mechanism。
>
>它决定每个 token representation 如何从其他 token positions 收集信息。

对于 autoregressive language model，attention 通常必须是 causal 的：

$$
h_t
\text{ only attends to }
h_{\leq t}
$$

这保证模型在预测下一个 token 时不能看到 future tokens。
### 4. Positional Information

Transformer 本身不天然知道 token 顺序，所以 architecture 必须规定 positional information 如何进入模型。

常见 choices：

- [[Absolute Positional Embedding]]
- [[Sinusoidal Positional Encoding]]
- [[Relative Position Information]]
- [[Rotary Position Embedding]]
- [[ALiBi]]

>[!question]
>为什么 position design 是 architecture 的一部分？
>
>因为它改变了模型表示 sequence order 的方式，也会影响 long-context behavior。

现代 LLM 中常见的是 [[Rotary Position Embedding]]。

直觉上：

- absolute position embedding 更像是给每个位置一个位置向量；
- RoPE 更像是让 position information 进入 attention computation；
- ALiBi 更像是给 attention score 加上和距离有关的 bias。

### 5. Normalization Design

Normalization design 决定模型如何稳定深层网络中的 hidden states。

常见 choices：

- [[Layer Normalization]]
- [[RMSNorm]]
- [[Pre-Norm Transformer]]
- [[Post-Norm Transformer]]

现代 LLM 中常见的是：

$$
\text{Pre-Norm} + \text{RMSNorm}
$$

>[!note]
>Normalization 不只是 training trick。
>
>Norm 的类型和位置改变了每一层 computation 的形式，所以它属于 architecture。

例如 Pre-Norm block 通常是：

$$
x + \mathrm{SubLayer}(\mathrm{Norm}(x))
$$

而 Post-Norm 是：

$$
\mathrm{Norm}(x + \mathrm{SubLayer}(x))
$$

两者的 gradient flow 和 training stability 不同。]]

### 6. MLP / FFN Design

每个 Transformer block 中通常包含一个 position-wise MLP / FFN。

Architecture 中的 MLP choices 包括：

- intermediate dimension 多大；
- FFN expansion ratio 是多少；
- activation function 是什么；
- 是否使用 gated activation；
- 是否使用 [[Mixture of Experts]]。

常见 activation choices：

- [[ReLU]]
- [[GELU]]
- [[SwiGLU]]

>[!note]
>在 Transformer block 中，attention 负责 token mixing，MLP 负责对每个 token representation 做 nonlinear processing。
>
>因此 MLP shape 和 activation function 都是 architecture choices。

现代 LLM 常见的是 gated MLP，例如 [[SwiGLU]]。
### 7. Residual and Block Layout

Transformer block 内部的连接方式也是 architecture 的一部分。

常见 choices：

- attention 和 MLP 是 sequential 还是 parallel；
- residual connection 如何加；
- norm 放在 residual branch 前还是后；
- dropout 是否出现在 block 内部；
- layer order 如何安排。

典型 Pre-Norm decoder-only block 可以写成：

$$
x'
=
x + \mathrm{Attention}(\mathrm{Norm}(x))
$$

$$
x_{\text{out}}
=
x' + \mathrm{MLP}(\mathrm{Norm}(x'))
$$

>[!note]
>Residual path 决定 information 和 gradient 如何穿过深层模型。
>
>这会影响训练稳定性，也会影响模型的实际 computation graph。
### 8. Output Design

Language model 最后需要把 hidden state 映射到 vocabulary logits。

$$
h_t^{(L)}
\mapsto
z_t \in \mathbb{R}^{V}
$$

其中 $V$ 是 vocabulary size。

常见 choices：

- output projection 是否和 token embedding weight tying；
- 是否使用 final normalization；
- vocabulary size 多大；
- logits 如何计算。

>[!note]
>Output projection 和 vocabulary logits 的设计属于 architecture。
>
>但 temperature、top-k、top-p 属于 Decoding Strategy，不是 model architecture。

## ⚖️ Architecture vs Hyperparameters

Architecture 和 hyperparameters 有重叠，但不完全等同。

| Concept | Meaning | Examples |
|---|---|---|
| Architecture | 模型 computation graph 的结构设计 | attention type, MLP type, norm placement, positional encoding |
| Model Hyperparameters | 控制 architecture size / shape 的具体数值 | number of layers, hidden dimension, number of heads, context length |
| Training Hyperparameters | 控制训练过程的数值 | learning rate, batch size, weight decay |

>[!note]
>$d_{\text{model}}$、$n_{\text{layers}}$、$n_{\text{heads}}$ 是 model hyperparameters。
>
>它们属于 architecture 的 shape choices。
>
>但 learning rate 和 batch size 不属于 architecture，而属于 [[Training Recipe]]。

## 🔄 Architecture vs Training Recipe

Architecture 定义模型结构；training recipe 定义模型如何被训练。

| Category                        | Belongs Here                                                                      | Does Not Belong Here                 |
| ------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------ |
| [[Model Architecture]]          | attention, [[MLP]], norm, position embedding, layer layout                        | learning rate, optimizer, batch size |
| [[Training Recipe]]             | [[Optimizer]], [[Learning Rate Schedule]], weight decay, batch size, data mixture | RoPE, RMSNorm, GQA                   |
| [[Systems for Language Models]] | memory layout, kernel optimization, parallelism, KV cache efficiency              | architecture 概念本身                    |

>[!important]
>同一个 architecture 可以用不同 training recipes 训练。
>
>不同 architectures 也可以用类似的 training recipe 训练。
>
>所以比较模型时，需要区分性能差异来自 architecture，还是来自 data / training / systems。

## 🦙 Example: Llama-style Architecture

[[Llama-style Architecture]] 是 modern decoder-only language model 的代表性 architecture pattern。

它通常包含：

| Component | Common Choice |
|---|---|
| overall family | [[Decoder-Only Transformer]] |
| attention | [[Causal Attention]] |
| position information | [[Rotary Position Embedding]] |
| normalization | [[RMSNorm]] |
| block layout | [[Pre-Norm Transformer]] |
| MLP activation | [[SwiGLU]] |
| attention variant | [[Grouped Query Attention]] |

>[!note]
>这说明 architecture 是一组组合选择，而不是一个单独标签。
>
>“Llama-style” 的意义在于：它把一组 modern LLM 中常见的 design choices 组合成了一个稳定范式。

---

>[!summary] My Understanding
>[[Model Architecture]] 描述模型内部 computation structure 的整体设计。
>
>它不只是“几层”或“多少参数”，而是包括：
>
>- overall model family；
>- depth / width / shape choices；
>- attention variant；
>- positional information；
>- normalization；
>- MLP / FFN design；
>- activation function；
>- residual and block layout；
>- output head。
>
>在 Lecture 3 的语境下，理解 architecture 的重点是：
>
>**现代 LLM 的能力和成本，不只由参数量决定，也由这些 architecture choices 共同决定。**

## 🔗 Connections

- [[Language Model Architecture]]
- [[Language Modeling]]
- [[Large Language Model (LLM)]]
- [[Transformer Family]]
- [[Decoder-Only Transformer]]
- [[Llama-style Architecture]]
- [[Transformer Block]]
- [[Self-Attention]]
- [[Causal Attention]]
- [[Artificial Intelligence/Transformer/Multi-Head Attention]] 
- [[Grouped Query Attention]]
- [[Rotary Position Embedding]]
- [[MLP]]
- [[Training Recipe]]
- [[Systems for Language Models]]
- [[Model Hyperparameters]]