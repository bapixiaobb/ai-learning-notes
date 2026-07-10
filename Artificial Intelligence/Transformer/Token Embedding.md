#LanguageModeling #Transformer #Embedding #Architecture

Token Embedding 指把 discrete token id 映射成 continuous vector 的方法。  
它是 language model 从离散文本进入神经网络计算的第一步。

## 🧠 Core Idea

>[!note]
>Language model 不能直接处理 raw text。
>
>文本会先经过 [[Tokenization]]，变成一串 token IDs；然后 Token Embedding 把每个 token ID 转成一个 dense vector。

例如：

```math
\text{``cat''}
\rightarrow
\text{token id}
\rightarrow
\text{embedding vector}
```

也可以写成：

```math
x_t
\mapsto
e_t \in \mathbb{R}^{d_{\text{model}}}
```

其中：

- $x_t$ 是第 $t$ 个 token id；
- $e_t$ 是对应的 embedding vector；
- $d_{\text{model}}$ 是 hidden dimension。

## 🔢 From Token ID to Vector

假设 vocabulary size 是 $V$，embedding dimension 是 $d_{\text{model}}$。

Embedding matrix 是：

```math
E \in \mathbb{R}^{V \times d_{\text{model}}}
```

每个 token id 对应 embedding matrix 里的一行：

```math
e_t = E[x_t]
```

>[!note]
>Token id 本身只是一个整数 index。
>
>真正进入 Transformer 的是它对应的 embedding vector。

## 🧩 Why Token Embedding Is Needed

Token IDs 是离散符号，例如：

```math
[15, 438, 92]
```

这些数字本身没有连续几何意义。  
模型需要把它们映射到 continuous vector space，才能进行 matrix multiplication、attention、MLP 等 neural network computation。

>[!important]
>Token Embedding 的作用不是表示 token 的“编号大小”，而是学习 token 在语义和使用模式上的向量表示。

例如，训练后某些语义或用法相近的 tokens 可能在 embedding space 中更接近。

## 📐 Shape View

对于一个 sequence：

```math
[x_1, x_2, \dots, x_T]
```

Token Embedding 后得到：

```math
X \in \mathbb{R}^{T \times d_{\text{model}}}
```

如果考虑 batch dimension：

```math
X \in \mathbb{R}^{B \times T \times d_{\text{model}}}
```

其中：

- $B$ 是 batch size；
- $T$ 是 sequence length；
- $d_{\text{model}}$ 是 hidden dimension。

这个 $X$ 就是 Transformer 的初始 hidden states / 初始 [[Residual Stream]]。

## 📍 Token Embedding vs Positional Encoding

[[Token Embedding]] 表示 token 是什么。  
[[Positional Encoding]] 表示 token 在哪里。

例如在句子中：

```math
\text{dog bites man}
```

Token Embedding 负责表示：

- dog；
- bites；
- man。

Positional Encoding 负责表示：

- dog 在第 1 个位置；
- bites 在第 2 个位置；
- man 在第 3 个位置。

>[!note]
>如果只有 Token Embedding，没有 positional information，Transformer 很难知道 token order。

## 🏗️ In Transformer

在 Transformer 中，输入通常先经过：

```math
\text{token IDs}
\rightarrow
\text{token embeddings}
\rightarrow
\text{positional information}
\rightarrow
\text{Transformer blocks}
```

在一些模型中，position information 会直接加到 token embeddings 上：

```math
h_t = e_t + p_t
```

在 modern LLM 中，position information 也可能通过 [[Rotary Position Embedding]] 进入 attention computation，而不是直接加到 embedding 上。

## 🔄 Learned Parameters

Embedding matrix 是模型参数的一部分，会在 training 中被学习。

也就是说：

```math
E
```

会随着 loss optimization 被更新。

>[!note]
>Token Embedding 不是固定词典解释。
>
>它是模型通过训练学到的 vector representation。

## 🔗 Input Embedding and Output Head

在 language model 中，输入端有 token embedding matrix：

```math
E \in \mathbb{R}^{V \times d_{\text{model}}}
```

输出端需要把 hidden states 映射回 vocabulary logits：

```math
z_t \in \mathbb{R}^{V}
```

有些模型会使用 [[Weight Tying]]，让 input embedding 和 output projection 共享权重。

>[!note]
>输入时，embedding 把 token id 映射到 hidden vector。
>
>输出时，language modeling head 把 hidden vector 映射回 vocabulary scores。

## 🚫 Common Confusions

### Token Embedding 不是 Tokenization

[[Tokenization]] 把 raw text 切成 token IDs。  
Token Embedding 把 token IDs 变成 vectors。

### Token ID 大小没有语义大小

token id 是 index，不是数值特征。  
token id 100 不代表比 token id 50 “更大”或“更重要”。

### Token Embedding 不包含完整上下文

初始 token embedding 主要表示 token identity。  
经过 [[Transformer Block]] 后，hidden states 才逐渐变成 contextual representations。

## 🧾 My Understanding

>[!summary]
>Token Embedding 是把 token id 映射成 dense vector 的方法。
>
>它让离散的 token symbols 进入 continuous vector space，从而可以被 Transformer 的 matrix multiplication、attention 和 MLP 处理。
>
>Token Embedding 表示 token 是什么；Positional Encoding 表示 token 在 sequence 中的位置。两者共同构成 Transformer 的初始输入表示。

## 🔗 Connections

- [[Tokenization]]
- [[Positional Encoding]]
- [[Rotary Position Embedding]]
- [[Transformer]]
- [[Transformer Block]]
- [[Residual Stream]]