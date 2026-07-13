#LanguageModeling #Architecture #Transformer #LLM

Llama-style Architecture 指 modern decoder-only language model 中非常常见的一组 architecture pattern。  
它不是说所有 LLM 都是 LLaMA，而是用 LLaMA 作为代表，理解现代开源 LLM 常用的结构组合。

## 🧠 Core Idea

>**Note**
>Llama-style Architecture 可以理解为：
>
>```math
>\text{Decoder-Only Transformer}
>+
>\text{modern architecture choices}
>```
>
>这些 choices 通常包括：
>
>- [Causal Attention](<./Causal%20Attention.md>)
>- [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>)
>- [RMSNorm](<./RMSNorm.md>)
>- [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>)
>- [SwiGLU](<./SwiGLU.md>)
>- Grouped Query Attention / Multi-Query Attention
>- Language Modeling Head

它的核心仍然是 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)：  
模型根据 prefix 预测下一个 token。

```math
p_\theta(x_t \mid x_{<t})
```

但相比 [Original Transformer](<./Original%20Transformer.md>)，modern LLM 在 normalization、position encoding、MLP activation、attention variant 等地方有很多不同。

## 🧭 Why “Llama-style”?

LLaMA 系列模型不是第一个 decoder-only Transformer，但它很好地代表了 modern open-source LLM 的常见结构。

在很多现代 LLM 中，经常看到类似组合：

| Design Area | Common Llama-style Choice |
|---|---|
| overall structure | [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>) |
| attention | [Causal Attention](<./Causal%20Attention.md>) |
| normalization | [RMSNorm](<./RMSNorm.md>) |
| norm placement | [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>) |
| position information | [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>) |
| MLP activation | [SwiGLU](<./SwiGLU.md>) |
| attention variant | Grouped Query Attention / Multi-Query Attention |
| objective | Next Token Prediction |

>**Important**
>“Llama-style” 不是一个严格数学定义，而是一种 modern decoder-only LLM 的 architecture shorthand。
>
>它强调的是一组常见 design choices 的组合，而不是某一个单独模块。

## 🧱 Relation to Original Transformer

[Original Transformer](<./Original%20Transformer.md>) 是 encoder-decoder architecture：

```math
\text{Encoder}
+
\text{Decoder}
```

而 Llama-style architecture 是 decoder-only：

```math
\text{Decoder-Only Transformer}
```

也就是说，它通常没有：

- encoder；
- encoder-decoder Cross-Attention；
- source-target sequence-to-sequence structure。

它保留并改造的是适合 autoregressive language modeling 的 causal self-attention 主线：

```math
x_{\leq t}
\rightarrow
p_\theta(x_{t+1} \mid x_{\leq t})
```

>**Note**
>所以 Llama-style model 不是 Original Transformer decoder 的简单复制。
>
>它是 decoder-only Transformer 在大规模 language modeling 语境下演化出的现代结构。

## 🧩 High-Level Pipeline

一个 Llama-style language model 的 forward pass 可以粗略写成：

```math
\text{token IDs}
\rightarrow
\text{token embeddings}
\rightarrow
\text{Transformer blocks}
\rightarrow
\text{final RMSNorm}
\rightarrow
\text{logits}
```

其中每个 Transformer block 通常包含：

```math
\text{RMSNorm}
\rightarrow
\text{Causal Attention with RoPE}
\rightarrow
\text{Residual Add}
\rightarrow
\text{RMSNorm}
\rightarrow
\text{SwiGLU MLP}
\rightarrow
\text{Residual Add}
```

更形式化地，可以写成：

```math
x'
=
x
+
\mathrm{Attention}(\mathrm{RMSNorm}(x))
```

```math
x_{\text{out}}
=
x'
+
\mathrm{SwiGLU\text{-}MLP}(\mathrm{RMSNorm}(x'))
```

这里的主 hidden state 流就是 [Residual Stream](<./Residual%20Stream.md>)。

## 🧵 Decoder-Only Structure

Llama-style architecture 属于 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)。

这意味着每个 token position 只能看到自己和之前的 tokens：

```math
h_t
\text{ can attend to }
h_{\leq t}
```

不能看到：

```math
h_{>t}
```

这个约束通过 [Causal Mask](<./Causal%20Mask.md>) 实现。

>**Note**
>Decoder-only structure 和 Next Token Prediction 天然匹配。
>
>模型在位置 $t$ 形成 prefix representation，然后用这个 representation 预测下一个 token。

这也是为什么 Llama-style architecture 适合 prompt-based generation：  
prompt 和 generated text 都可以被看成同一个 autoregressive token stream。

## 🎭 Causal Attention

Llama-style model 使用 [Causal Attention](<./Causal%20Attention.md>)。

Self-attention 的基本形式仍然是：

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
+
M
\right)V
```

其中 $M$ 是 causal mask，用来屏蔽 future positions。

>**Note**
>Attention 的作用是 token mixing。
>
>在 causal setting 下，每个 token 只能从 prefix 中收集信息，因此不会偷看未来 token。

在现代 LLM 中，attention 还经常和 [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>)、Grouped Query Attention、KV Cache 结合。

## 📍 Rotary Position Embedding

Llama-style architecture 通常使用 [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>) 表示位置信息。

RoPE 的特点是：

- position information 不是简单加到 token embedding 上；
- 它作用在 attention 中的 $Q$ 和 $K$ 上；
- 它让 attention score 感知相对位置信息。

>**Note**
>[Original Transformer](<./Original%20Transformer.md>) 使用 sinusoidal positional encoding，加到 token embeddings 上。
>
>Llama-style model 更常使用 RoPE，把 position information 融入 attention computation。

因此，在 Llama-style block 中，position information 通常出现在 attention branch 里，而不是作为单独的 additive embedding。

## ⚖️ RMSNorm

Llama-style architecture 通常使用 [RMSNorm](<./RMSNorm.md>)，而不是 standard [Layer Normalization](<./Layer%20Normalization.md>)。

RMSNorm 的直觉是：  
它主要根据 root mean square 来缩放 hidden vector，而不像 LayerNorm 那样显式减去 mean。

>**Note**
>RMSNorm 是 normalization choice。
>
>它属于 [Model Architecture](<../Language%20modeling/05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>)，因为它改变了每个 block 内部 hidden states 的处理方式。

现代 LLM 常用 RMSNorm 的原因包括：

- 计算更简单；
- 与 Pre-Norm block 搭配常见；
- 在大规模 decoder-only LLM 中表现稳定。

## 🔁 Pre-Norm Transformer

Llama-style architecture 通常采用 [Pre-Norm Transformer](<./Pre-Norm%20Transformer.md>)。

Pre-Norm 的形式是：

```math
x_{\text{out}}
=
x
+
F(\mathrm{Norm}(x))
```

而不是 Post-Norm：

```math
x_{\text{out}}
=
\mathrm{Norm}(x + F(x))
```

>**Note**
>[Original Transformer](<./Original%20Transformer.md>) 使用的是 Post-Norm。
>
>Modern LLM 更常用 Pre-Norm，因为它通常更适合训练深层 Transformer。

在 Llama-style block 中，attention 和 MLP 前面都会先做 RMSNorm：

```math
x'
=
x
+
\mathrm{Attention}(\mathrm{RMSNorm}(x))
```

```math
x_{\text{out}}
=
x'
+
\mathrm{MLP}(\mathrm{RMSNorm}(x'))
```

## 🧮 SwiGLU MLP

Llama-style architecture 通常使用 [SwiGLU](<./SwiGLU.md>) 作为 MLP activation / gated MLP。

普通 Transformer FFN 可以粗略写成：

```math
\mathrm{FFN}(x)
=
W_2 \sigma(W_1 x)
```

而 SwiGLU-style MLP 通常包含 gating 结构：

```math
\mathrm{SwiGLU}(x)
=
\mathrm{SiLU}(xW_1)
\odot
(xW_3)
```

然后再投影回 hidden dimension：

```math
\mathrm{MLP}(x)
=
\mathrm{SwiGLU}(x)W_2
```

>**Note**
>MLP 负责 per-token nonlinear processing。
>
>SwiGLU 通过 gating mechanism 增强 MLP 的表达能力，是 modern LLM 中很常见的 architecture choice。

这里的 $\odot$ 是 element-wise product。

## 🧠 Attention Variant: MHA, MQA, GQA

Llama-style architecture 中的 attention variant 可能不是最原始的 [Multi-Head Attention](<./Multi-Head%20Attention.md>)。

常见 choices 包括：

| Variant | Idea |
|---|---|
| [Multi-Head Attention](<./Multi-Head%20Attention.md>) | 每个 query head 有自己的 key/value head |
| Multi-Query Attention | 多个 query heads 共享一组 key/value |
| Grouped Query Attention | 多个 query heads 分组共享 key/value |

>**Note**
>Grouped Query Attention 经常用于 modern LLM，因为它在保留多头 query 表达能力的同时，可以减少 KV cache 的 memory cost。

这里可以看到 architecture 和 systems 的联系：  
GQA 是 architecture choice，但它会直接影响 KV Cache 和 Inference Cost。

## 🧱 Llama-style Transformer Block

一个 Llama-style block 可以理解成：

```math
\text{Residual Stream}
\rightarrow
\text{RMSNorm}
\rightarrow
\text{Causal Attention + RoPE}
\rightarrow
\text{Residual Add}
\rightarrow
\text{RMSNorm}
\rightarrow
\text{SwiGLU MLP}
\rightarrow
\text{Residual Add}
```

对应公式：

```math
a
=
\mathrm{CausalAttention}
\left(
\mathrm{RMSNorm}(x)
\right)
```

```math
x'
=
x + a
```

```math
m
=
\mathrm{SwiGLUMLP}
\left(
\mathrm{RMSNorm}(x')
\right)
```

```math
x_{\text{out}}
=
x' + m
```

>**Important**
>这个 block 的核心数据流仍然是 [Residual Stream](<./Residual%20Stream.md>)。
>
>Attention 和 MLP 都是在 residual stream 上添加更新量，而不是完全替换 hidden states。

## 🎯 Language Modeling Head

经过 $L$ 个 Transformer blocks 后，最终 hidden states 会通过 final norm 和 output projection 得到 logits：

```math
H^{(L)}
\rightarrow
\mathrm{RMSNorm}
\rightarrow
Z
```

其中：

```math
Z
\in
\mathbb{R}^{T \times V}
```

$V$ 是 vocabulary size。

每个位置的 logits 用来预测下一个 token：

```math
p_\theta(x_{t+1} \mid x_{\leq t})
=
\mathrm{softmax}(z_t)
```

>**Note**
>Logits 是未归一化分数，不是 probability。
>
>经过 [Softmax](<./Softmax.md>) 后才得到 next-token probability distribution。

## ⚙️ Architecture vs Training Recipe

Llama-style architecture 描述的是模型结构，不是训练方案。

例如：

| Belongs to Architecture | Belongs to Training Recipe  |
| ----------------------- | --------------------------- |
| RMSNorm                 | [Optimizer](<./Optimizer.md>)               |
| RoPE                    | [Learning Rate Schedule](<./Learning%20Rate%20Schedule.md>)  |
| SwiGLU                  | batch size                  |
| causal attention        | weight decay                |
| GQA                     | data mixture                |
| decoder-only structure  | training tokens             |

>**Note**
>两个模型可以采用相似的 Llama-style architecture，但因为 training data、training tokens、[Optimizer](<./Optimizer.md>)、[Learning Rate Schedule](<./Learning%20Rate%20Schedule.md>) 不同，最终能力差异很大。
>
>因此比较 LLM 时，不能只看 architecture，也要看 [Training Recipe](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Training%20Recipe.md>)。

## 🧊 Systems Implications

Llama-style architecture 中的一些 choices 会直接影响 systems cost。

例如：

- context length 影响 attention cost；
- GQA / MQA 影响 KV Cache memory；
- MLP hidden size 影响 FLOPs；
- number of layers 影响 latency；
- hidden dimension 影响 parameter count 和 activation memory。

>**Note**
>Architecture choices 不等于 systems optimization，但它们会决定很多 compute / memory bottlenecks。
>
>这部分连接到 Systems for Language Models 和 [Resource Accounting](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)。

---

>**Summary** — My Understanding
>Llama-style Architecture 是 modern decoder-only LLM 的一种典型结构组合。
>
>它的核心仍然是 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>) 和 Next Token Prediction，但在 block 内部采用了一组现代 architecture choices：
>
>- [RMSNorm](<./RMSNorm.md>)
>- [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>)
>- [SwiGLU](<./SwiGLU.md>)
>- [Causal Attention](<./Causal%20Attention.md>)
>- Grouped Query Attention
>
>理解 Llama-style architecture 的关键是：
>
>**它不是 Original Transformer 的完整 encoder-decoder 结构，而是 modern decoder-only language model 的常见范式。**

## 🔗 Connections

- [Language Model Architecture](<../Language%20modeling/05%20-%20Architectures%20and%20MoE/Language%20Model%20Architecture.md>)
- [Model Architecture](<../Language%20modeling/05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>)
- [Transformer](<./Transformer.md>)
- [Transformer Family](<./Transformer%20Family.md>)
- [Original Transformer](<./Original%20Transformer.md>)
- [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Residual Stream](<./Residual%20Stream.md>)
- [Residual Connection](<./Residual%20Connection.md>)
- [RMSNorm](<./RMSNorm.md>)
- [Layer Normalization](<./Layer%20Normalization.md>)
- [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>)
- [MLP](<./MLP.md>)
- Grouped Query Attention
- Multi-Query Attention
- KV Cache
- Logits
- [Softmax](<./Softmax.md>)
- Next Token Prediction
- [Training Recipe](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Training%20Recipe.md>)
- Systems for Language Models
- [Resource Accounting](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)