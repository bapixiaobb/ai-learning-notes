#DeepLearning #NeuralNetwork #NLP #LanguageModeling #Architecture #Transformer

Decoder-Only Transformer 是只保留 Transformer decoder 主干的一类 Transformer architecture。  
现代 autoregressive LLM，例如 GPT / LLaMA-style models，大多采用 decoder-only Transformer。

## 🧠 Core Idea

>[!note]
>[[Decoder-Only Transformer]] 的核心是：
>
>用 causal self-attention 建模 token prefix，并根据 prefix 预测下一个 token。
>
>它没有 encoder，也通常没有 encoder-decoder Cross-Attention。

它建模的是 autoregressive factorization：

$$
p_\theta(x_1, x_2, \dots, x_T)
=
\prod_{t=1}^{T}
p_\theta(x_t \mid x_{<t})
$$

也就是说，模型在每个位置只能基于过去的 tokens 预测当前 / 下一个 token。

## 🧭 Why “Decoder-Only”?

“Decoder-only” 这个名字来自 [[Original Transformer]] 的 encoder-decoder 结构。

Original Transformer 包含：

$$
\text{Encoder}
+
\text{Decoder}
$$

而 decoder-only Transformer 可以理解为：

$$
\text{Transformer Decoder}
-
\text{Encoder}
-
\text{Cross-Attention}
$$

保留下来的主要是：

- masked / causal self-attention；
- MLP / FFN；
- residual connection；
- normalization；
- language modeling output head。

>[!important]
>现代 GPT / LLaMA-style model 虽然叫 decoder-only，但它不等于直接复制 Original Transformer 的 decoder。
>
>它通常去掉了 encoder-decoder cross-attention，并采用 modern LLM 的 architecture choices，例如 [[RMSNorm]]、[[Rotary Position Embedding]]、[[SwiGLU]]、[[Pre-Norm Transformer]] 等。

## 🧱 Relation to Original Transformer

在 [[Original Transformer]] 中，decoder block 通常包含三部分：

1. masked self-attention；
2. encoder-decoder cross-attention；
3. feed-forward network。

而 [[Decoder-Only Transformer]] 通常只保留：

1. causal self-attention；
2. feed-forward / MLP；
3. residual + normalization。

可以粗略对比：

| Component                      | Original Transformer Decoder | Decoder-Only Transformer  |
| ------------------------------ | ---------------------------- | ------------------------- |
| masked / causal self-attention | yes                          | yes                       |
| cross-attention to encoder     | yes                          | usually no                |
| encoder outputs                | yes                          | no                        |
| FFN / MLP                      | yes                          | yes                       |
| residual connection            | yes                          | yes                       |
| normalization                  | yes                          | yes                       |
| output head                    | target vocabulary prediction | [[Next-token prediction]] |

>[!note]
>所以 decoder-only 的关键不是“只用 decoder 这张图”，而是：
>
>它只保留适合 autoregressive generation 的 causal self-attention 主线。

## 🧩 Basic Pipeline

一个 decoder-only language model 的基本数据流可以写成：

$$
\text{tokens}
\rightarrow
\text{token embeddings}
\rightarrow
\text{positional information}
\rightarrow
\text{decoder-only Transformer blocks}
\rightarrow
\text{final hidden states}
\rightarrow
\text{logits}
$$

对于输入 token sequence：

$$
x_1, x_2, \dots, x_T
$$

模型输出每个位置的 logits：

$$
z_1, z_2, \dots, z_T
$$

其中：

$$
z_t \in \mathbb{R}^{V}
$$

$V$ 是 vocabulary size。

通常 $z_t$ 用来预测下一个 token：

$$
p_\theta(x_{t+1} \mid x_{\leq t})
=
\operatorname{softmax}(z_t)
$$

## 🎭 Causal Self-Attention

Decoder-only Transformer 的关键是 [[Causal Attention]]。

在第 $t$ 个位置，模型只能 attend to 当前和之前的位置：

$$
h_t
\text{ can attend to }
h_1, h_2, \dots, h_t
$$

不能 attend to：

$$
h_{t+1}, h_{t+2}, \dots, h_T
$$

这通过 [[Causal Mask]] 实现。

>[!note]
>Causal mask 保证模型不会偷看 future tokens。
>
>这使 decoder-only Transformer 和 [[Next Token Prediction]] 的训练目标匹配。

注意这里的 “decoder” 不是在说模型一定有 encoder-decoder translation setting。  
在 GPT / LLaMA-style language model 中，整个输入都被看作一个 autoregressive token stream。

## 🏗️ Decoder-Only Transformer Block

一个 decoder-only Transformer block 通常包含：

- causal self-attention；
- MLP / FFN；
- normalization；
- residual connection。

现代 LLM 中常见的 Pre-Norm 形式可以写成：

$$
x'
=
x + \mathrm{CausalAttention}(\mathrm{Norm}(x))
$$

$$
x_{\text{out}}
=
x' + \mathrm{MLP}(\mathrm{Norm}(x'))
$$

>[!note]
>[[Transformer Block]] 是 decoder-only Transformer 的核心重复单元。
>
>多个 blocks 堆叠后，模型逐层更新每个 token position 的 hidden representation。

这条主 hidden state 流动路径经常被称为 [[Residual Stream]]。

## 🔁 Training Data Flow

在训练 decoder-only language model 时，通常把一个 token sequence 同时作为 input 和 target，只是 target 相对 input shift 一位。

例如输入：

$$
[x_1, x_2, x_3, x_4]
$$

模型在各位置预测：

$$
[x_2, x_3, x_4, x_5]
$$

也就是：

| Position | Visible Prefix | Prediction Target |
|---|---|---|
| 1 | $x_1$ | $x_2$ |
| 2 | $x_1, x_2$ | $x_3$ |
| 3 | $x_1, x_2, x_3$ | $x_4$ |
| 4 | $x_1, x_2, x_3, x_4$ | $x_5$ |

>[!note]
>训练时所有 positions 可以并行计算 loss。
>
>虽然模型在每个位置只能看到 prefix，但矩阵计算可以一次性处理整个 sequence；causal mask 负责屏蔽 future positions。

这就是为什么 decoder-only Transformer 同时适合 autoregressive modeling 和大规模并行训练。

## 🧮 Inference Data Flow

推理时，decoder-only Transformer 通常 autoregressively 生成 token。

给定 prompt：

$$
x_1, x_2, \dots, x_T
$$

模型先预测：

$$
p_\theta(x_{T+1} \mid x_{\leq T})
$$

采样或选择一个 token 后，再把它接到序列后面：

$$
x_1, x_2, \dots, x_T, x_{T+1}
$$

继续预测：

$$
p_\theta(x_{T+2} \mid x_{\leq T+1})
$$

>[!note]
>训练时可以并行计算多个 positions 的 loss；推理时生成过程通常是 sequential 的。
>
>这就是 [[Training vs Inference]] 在 decoder-only Transformer 中最重要的区别之一。

为了避免每一步重复计算所有 previous tokens，推理时通常使用 [[KV Cache]]。

## 🔢 Attention Mask Shape

如果 sequence length 是 $T$，causal mask 可以理解成一个下三角矩阵：

$$
M =
\begin{bmatrix}
1 & 0 & 0 & 0 \\
1 & 1 & 0 & 0 \\
1 & 1 & 1 & 0 \\
1 & 1 & 1 & 1
\end{bmatrix}
$$

第 $t$ 行表示第 $t$ 个 token 可以 attend to 哪些 positions。

>[!note]
>这个 mask 的含义是：
>
>第 $t$ 个位置只能看见 $1,\dots,t$，不能看见 $t+1,\dots,T$。

在实际实现中，mask 通常加到 attention scores 上，把 future positions 的 score 变成 $-\infty$，使 [[Softmax]] 后权重为 0。

## 🦙 Llama-style Decoder-Only Transformer

现代 LLM 中常见的是 Llama-style decoder-only Transformer。

它通常包含：

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
>Llama-style architecture 可以理解为 modern decoder-only Transformer 的一组 common design choices。
>
>它不是 Original Transformer decoder 的简单复制，而是经过多次 architecture evolution 后形成的现代结构范式。

## ⚖️ Decoder-Only vs Encoder-Only vs Encoder-Decoder

| Architecture                    | Attention Pattern                                        | Typical Objective         | Typical Use                             |
| ------------------------------- | -------------------------------------------------------- | ------------------------- | --------------------------------------- |
| [[Encoder-Only Transformer]]    | bidirectional self-attention                             | masked language modeling  | representation learning, classification |
| [[Decoder-Only Transformer]]    | causal self-attention                                    | [[Next-token prediction]] | autoregressive generation, LLMs         |
| [[Encoder-Decoder Transformer]] | encoder bidirectional + decoder causal + cross-attention | conditional generation    | translation, summarization              |

>[!important]
>Decoder-only Transformer 适合 generation，因为它的 causal structure 和 autoregressive factorization 一致。
>
>Encoder-only Transformer 更适合理解输入；encoder-decoder Transformer 更适合 source-to-target conditional generation。

## 🧠 Why Modern LLMs Use Decoder-Only Transformer

现代 LLM 大多使用 decoder-only Transformer，主要因为它和大规模 autoregressive pretraining 很匹配。

它的优势包括：

- 训练目标简单：[[Next-token prediction]]；
- 数据形式统一：任意 text 都可以看成 token stream；
- inference 形式自然：不断生成下一个 token；
- architecture 相对简洁：没有 encoder 和 cross-attention；
- scale-up 路径清晰：增加 layers、width、data、compute；
- 可以通过 prompt 把很多任务统一成 text continuation。

>[!note]
>Decoder-only Transformer 的强大之处在于：
>
>它把很多 NLP tasks 都转化成了同一种形式：
>
>$$
>\text{given prefix} \rightarrow \text{predict continuation}
>$$

---

>[!summary] My Understanding
>[[Decoder-Only Transformer]] 是 modern autoregressive LLM 的主流 architecture。
>
>它来自 [[Original Transformer]] 的 decoder side，但通常去掉了 encoder 和 cross-attention，只保留 causal self-attention + MLP + residual + normalization 的主干。
>
>它的核心约束是：每个 token position 只能看见 prefix，因此天然适合 [[Next Token Prediction]]。
>
>GPT / LLaMA-style models 可以理解为 decoder-only Transformer 加上一组 modern architecture choices，例如 [[RMSNorm]]、[[Rotary Position Embedding]]、[[SwiGLU]] 和 [[Grouped Query Attention]]。

## 🔗 Connections

- [[Transformer]]
- [[Transformer Family]]
- [[Original Transformer]]
- [[Encoder-Only Transformer]] 
- [[Llama-style Architecture]]
- [[Language Model Architecture]]
- [[Model Architecture]]
- [[Self-Attention]]
- [[Transformer Block]]
- [[Residual Stream]]
- [[MLP]]
- [[Normalization]]
- [[RMSNorm]]
- [[Rotary Position Embedding]]
- [[Grouped Query Attention]]
- [[KV Cache]]
- [[Training vs Inference]]