#DeepLearning #NeuralNetwork #Transformer #Architecture #LanguageModeling

Forward Pass in Transformer 指输入 tokens 经过 Transformer 的各个模块，最终得到 hidden states 或 logits 的计算过程。  
它描述的是模型在一次前向计算中，数据如何从输入流向输出。

## 🧠 Core Idea

>**Note**
>Forward pass 不是“只走一遍 attention”。
>
>在 Transformer 中，forward pass 通常包括：
>
>- [Token Embedding](<./Token%20Embedding.md>)
>- [Positional Encoding](<./Positional%20Encoding.md>) / [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>)
>- 多层 [Transformer Block](<./Transformer%20Block.md>)
>- [Self-Attention](<./Self-Attention.md>)
>- [MLP](<./MLP.md>)
>- [Residual Connection](<./Residual%20Connection.md>)
>- [Normalization](<./Normalization.md>)
>- final hidden states
>- Language Modeling Head
>- Logits

对于 language model，forward pass 可以粗略写成：

```math
\text{token IDs}
\rightarrow
\text{embeddings}
\rightarrow
\text{Transformer blocks}
\rightarrow
\text{hidden states}
\rightarrow
\text{logits}
```

如果是在训练阶段，还会继续计算 loss；如果是在推理阶段，则会根据 logits 选择或采样下一个 token。

## 🧭 What Forward Pass Means

Forward pass 是模型根据当前参数 $\theta$ 计算输出的过程。

给定输入：

```math
x_1, x_2, \dots, x_T
```

Transformer 会计算：

```math
f_\theta(x_1, x_2, \dots, x_T)
```

在 language model 中，输出通常是每个位置的 logits：

```math
Z
\in
\mathbb{R}^{T \times V}
```

其中：

- $T$ 是 sequence length；
- $V$ 是 vocabulary size；
- $Z_t$ 是第 $t$ 个位置对应的 vocabulary logits。

>**Note**
>Forward pass 本身不更新参数。
>
>参数更新发生在 backward pass 和 [Optimizer](<./Optimizer.md>) step 中。Forward pass 只是在当前参数下计算 activations、hidden states、logits 和 loss。

## 🧩 High-Level Pipeline

一个 decoder-only language model 的 forward pass 可以写成：

```math
\text{raw text}
\rightarrow
\text{tokenization}
\rightarrow
\text{token IDs}
\rightarrow
\text{token embeddings}
\rightarrow
\text{initial residual stream}
\rightarrow
\text{Transformer blocks}
\rightarrow
\text{final residual stream}
\rightarrow
\text{logits}
```

如果输入 token IDs 是：

```math
X =
[x_1, x_2, \dots, x_T]
```

则 token embedding 后得到：

```math
H^{(0)}
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

然后经过 $L$ 层 Transformer blocks：

```math
H^{(0)}
\rightarrow
H^{(1)}
\rightarrow
\cdots
\rightarrow
H^{(L)}
```

最后通过 output head 得到：

```math
Z
=
H^{(L)} W_{\text{out}}
```

其中：

```math
Z
\in
\mathbb{R}^{T \times V}
```

## 🧱 Step 1: Token IDs to Embeddings

Transformer 的输入不是 raw text，而是 token IDs。

例如：

```math
[x_1, x_2, \dots, x_T]
```

其中每个 $x_t$ 是 vocabulary 中的一个 integer index。

[Token Embedding](<./Token%20Embedding.md>) 会把每个 token id 映射成 dense vector：

```math
x_t
\mapsto
e_t
\in
\mathbb{R}^{d_{\text{model}}}
```

整个 sequence 变成：

```math
E
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

>**Note**
>从这一刻开始，Transformer 处理的是 continuous hidden vectors，而不是离散的 text。

如果考虑 batch dimension，则 shape 是：

```math
B \times T \times d_{\text{model}}
```

## 📍 Step 2: Add or Apply Positional Information

Transformer 的 self-attention 本身不天然知道 token 顺序，所以需要 positional information。

在 [Original Transformer](<./Original%20Transformer.md>) 中，position information 通常通过 positional encoding 加到 token embeddings 上：

```math
H^{(0)}
=
E
+
P
```

其中：

- $E$ 是 token embeddings；
- $P$ 是 positional encodings。

在 modern LLM 中，常见的是 [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>)。  
RoPE 通常不是简单加到 embedding 上，而是在 attention 计算中作用到 $Q$ 和 $K$。

>**Note**
>无论具体方式是什么，positional information 都是 forward pass 中让模型理解 sequence order 的关键部分。

## 🧵 Step 3: Initial Residual Stream

embedding 之后得到的 hidden states 可以看作初始 [Residual Stream](<./Residual%20Stream.md>)：

```math
H^{(0)}
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

这个 residual stream 会被送入第一层 [Transformer Block](<./Transformer%20Block.md>)。

随后每个 block 都会读取当前 residual stream，并写入新的更新量：

```math
H^{(\ell+1)}
=
H^{(\ell)}
+
\Delta^{(\ell)}
```

>**Note**
>Residual stream 是贯穿 forward pass 的主 hidden state 路径。
>
>Attention 和 MLP 都不是单独“另开一条最终输出路径”，而是不断更新这条主数据流。

## 🏗️ Step 4: Forward Through Transformer Blocks

Transformer 的主体是多个 [Transformer Block](<./Transformer%20Block.md>)。

一个 modern Pre-Norm decoder-only block 可以写成：

```math
A^{(\ell)}
=
\mathrm{Attention}
\left(
\mathrm{Norm}(H^{(\ell)})
\right)
```

```math
\tilde{H}^{(\ell)}
=
H^{(\ell)}
+
A^{(\ell)}
```

然后：

```math
M^{(\ell)}
=
\mathrm{MLP}
\left(
\mathrm{Norm}(\tilde{H}^{(\ell)})
\right)
```

```math
H^{(\ell+1)}
=
\tilde{H}^{(\ell)}
+
M^{(\ell)}
```

也就是说，一个 block 内部通常有两次主要更新：

1. attention update；
2. MLP update。

>**Important**
>Forward pass 不是：
>
>```math
>\text{input} \rightarrow \text{attention} \rightarrow \text{output}
>```
>
>而是：
>
>```math
>\text{residual stream}
>\rightarrow
>\text{attention update}
>\rightarrow
>\text{residual stream}
>\rightarrow
>\text{MLP update}
>\rightarrow
>\text{residual stream}
>```

## 🎭 Attention Branch

在 attention branch 中，当前 hidden states 会被投影成 $Q,K,V$：

```math
Q = HW_Q
```

```math
K = HW_K
```

```math
V = HW_V
```

然后计算 attention scores：

```math
\frac{QK^\top}{\sqrt{d_k}}
```

经过 mask 和 [Softmax](<./Softmax.md>) 后，对 $V$ 做加权求和，得到 attention output。

在 [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>) 中，attention 通常是 [Causal Attention](<./Causal%20Attention.md>)，所以第 $t$ 个位置只能看见：

```math
1,2,\dots,t
```

不能看见：

```math
t+1,t+2,\dots,T
```

>**Note**
>Attention branch 的作用是 token mixing。
>
>它让每个 token position 从允许看到的其他 positions 收集上下文信息。

Attention output 会通过 [Residual Connection](<./Residual%20Connection.md>) 加回 residual stream，而不是直接成为整个模型的最终输出。

## 🧮 MLP Branch

在 MLP branch 中，每个 token position 独立经过 nonlinear transformation。

对于某个 token hidden vector：

```math
h_t
\in
\mathbb{R}^{d_{\text{model}}}
```

MLP 通常做：

```math
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
```

也就是：

```math
\mathrm{MLP}(h_t)
=
W_2 \sigma(W_1 h_t)
```

现代 LLM 中可能使用 [SwiGLU](<./SwiGLU.md>) 这样的 gated MLP。

>**Note**
>MLP branch 不负责 token positions 之间的信息交换。
>
>它对每个 position 独立作用，负责 nonlinear feature processing。

MLP output 同样会加回 residual stream：

```math
H^{(\ell+1)}
=
\tilde{H}^{(\ell)}
+
M^{(\ell)}
```

## 🔁 Multiple Layers

经过一个 block 后，hidden states 仍然保持同样 shape：

```math
T \times d_{\text{model}}
\rightarrow
T \times d_{\text{model}}
```

所以可以继续送入下一层：

```math
H^{(0)}
\rightarrow
H^{(1)}
\rightarrow
H^{(2)}
\rightarrow
\cdots
\rightarrow
H^{(L)}
```

随着层数加深，每个 token position 的 representation 会逐渐融合更多上下文信息。

在 decoder-only language model 中，第 $t$ 个位置的 hidden state 只能依赖 prefix：

```math
x_{\leq t}
```

不能依赖 future tokens：

```math
x_{>t}
```

这一点由 [Causal Mask](<./Causal%20Mask.md>) 保证。

## 🎯 Step 5: Final Hidden States to Logits

最后一层输出：

```math
H^{(L)}
\in
\mathbb{R}^{T \times d_{\text{model}}}
```

会通过 Language Modeling Head 映射到 vocabulary logits：

```math
Z
=
H^{(L)} W_{\text{out}}
```

其中：

```math
W_{\text{out}}
\in
\mathbb{R}^{d_{\text{model}} \times V}
```

所以：

```math
Z
\in
\mathbb{R}^{T \times V}
```

每个位置 $t$ 的 logits：

```math
z_t
\in
\mathbb{R}^{V}
```

表示模型对 vocabulary 中每个 token 的未归一化分数。

>**Note**
>Logits 不是 probability。
>
>需要经过 [Softmax](<./Softmax.md>) 才能得到 next-token probability distribution。

## 📉 Step 6: Loss During Training

训练时，forward pass 通常还包括 loss computation。

对于 decoder-only language model，位置 $t$ 的 logits 通常用于预测下一个 token：

```math
p_\theta(x_{t+1} \mid x_{\leq t})
=
\mathrm{softmax}(z_t)
```

loss 通常是 cross-entropy：

```math
\mathcal{L}
=
-
\sum_t
\log p_\theta(x_{t+1} \mid x_{\leq t})
```

>**Note**
>Loss 是 forward pass 计算出的标量结果。
>
>但参数不会在 forward pass 中更新。参数更新需要 backward pass 计算 gradients，然后由 optimizer step 完成。

这部分属于 [Training Recipe](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Training%20Recipe.md>)，而不是 [Model Architecture](<../Language%20modeling/05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>) 本身。

## 🚀 Forward Pass During Inference

推理时，forward pass 用来产生下一个 token 的 logits。

给定 prompt：

```math
x_1, x_2, \dots, x_T
```

模型 forward 一次得到最后位置的 logits：

```math
z_T
```

然后通过 decoding strategy 选择下一个 token：

```math
x_{T+1}
```

再把它接到 sequence 后面继续生成：

```math
x_1, x_2, \dots, x_T, x_{T+1}
```

>**Note**
>推理阶段也有 forward pass。
>
>区别是：推理时通常不计算 training loss，也不做 backward pass 和 optimizer update。

为了避免重复计算历史 tokens，推理时通常使用 KV Cache。

## ⚖️ Training Forward vs Inference Forward

| Aspect | Training Forward | Inference Forward |
|---|---|---|
| input | training sequences | prompt + generated tokens |
| output | logits + loss | logits for next token |
| target labels | yes | no |
| backward pass | usually yes | no |
| parameter update | after backward | no |
| causal mask | yes in decoder-only LM | yes in decoder-only LM |
| KV cache | usually not the main mode | commonly used |

>**Important**
>Training 和 inference 都会发生 forward pass。
>
>区别不在于有没有 forward，而在于 forward 之后是否计算 loss、是否 backward、是否更新参数，以及是否使用 KV cache 做高效 generation。

## 🔄 Forward Pass vs Backward Pass

Forward pass 计算：

```math
\text{inputs}
\rightarrow
\text{activations}
\rightarrow
\text{logits}
\rightarrow
\text{loss}
```

Backward pass 计算：

```math
\frac{\partial \mathcal{L}}{\partial \theta}
```

也就是 loss 对模型参数的 gradients。

Optimizer step 使用这些 gradients 更新参数：

```math
\theta
\leftarrow
\theta
-
\eta \nabla_\theta \mathcal{L}
```

>**Note**
>Forward pass 是“算输出”。
>
>Backward pass 是“算梯度”。
>
>Optimizer step 是“改参数”。

这三个过程都属于训练循环的一部分，但概念不同。

## 🧠 Common Confusions

### 1. Forward pass 不是只走 attention

一个完整 Transformer block 通常同时包含 attention branch 和 MLP branch：

```math
\text{Attention}
+
\text{MLP}
+
\text{Residual Connection}
+
\text{Normalization}
```

所以 forward pass 不只是 attention computation。

### 2. Forward pass 不等于 inference

Inference 一定包含 forward pass，但 forward pass 不只发生在 inference。  
训练时也需要 forward pass 来计算 logits 和 loss。

### 3. Forward pass 不更新参数

Forward pass 只计算输出和中间 activations。  
参数更新发生在 backward pass 和 optimizer step。

### 4. Attention output 不是最终 logits

Attention output 会加回 [Residual Stream](<./Residual%20Stream.md>)，再经过后续 MLP、后续 layers、final norm 和 output head，最终才得到 logits。

---

>**Summary** — My Understanding
>Forward Pass in Transformer 描述的是输入 tokens 如何经过 Transformer 被计算成 hidden states 和 logits。
>
>在 decoder-only language model 中，forward pass 从 token embeddings 开始，形成初始 residual stream；每个 Transformer block 通过 attention branch 和 MLP branch 更新 residual stream；最后的 hidden states 经过 language modeling head 得到 logits。
>
>Forward pass 既发生在 training 中，也发生在 inference 中。它本身不更新参数；参数更新属于 backward pass 和 optimizer step。
>
>理解 forward pass 的关键是：
>
>**Transformer 的数据流不是只经过 attention，而是沿着 residual stream 被 attention 和 MLP 反复更新。**

## 🔗 Connections

- [Transformer](<./Transformer.md>)
- [Transformer Block](<./Transformer%20Block.md>)
- [Decoder-Only Transformer](<./Decoder-Only%20Transformer.md>)
- [Residual Stream](<./Residual%20Stream.md>)
- [Residual Connection](<./Residual%20Connection.md>)
- [Self-Attention](<./Self-Attention.md>)
- [MLP](<./MLP.md>)
- [Normalization](<./Normalization.md>)
- [Token Embedding](<./Token%20Embedding.md>)
- [Positional Encoding](<./Positional%20Encoding.md>)
- [Rotary Position Embedding](<./Rotary%20Position%20Embedding.md>)
- [Training vs Inference](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Training%20vs%20Inference.md>)
- [Training Recipe](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Training%20Recipe.md>)
- KV Cache