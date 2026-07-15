#LanguageModeling #DeepLearning #LinearAlgebra

**定义**：Embedding 是把离散对象映射成连续向量表示的过程。在 language model 中，embedding 通常指把 token ID 映射成一个向量。

```math
\text{token ID} \rightarrow \mathbf{x} \in \mathbb{R}^{d_{\text{model}}}
```

其实就是一个查表工作，拿到 ID 去找 Embedding table 对应的 vocabulary row
## 直觉

Token ID 本身只是一个整数，例如：

```text
"cat" -> 5234
```

这个数字没有直接的数学语义。Embedding table 会把每个 token ID 映射成一个可学习的向量：

```math
5234 \rightarrow \mathbf{x}_{5234}
```

这些向量会在训练过程中被更新，使语义或用法相近的 tokens 在向量空间中更接近。

## **数学形式**

假设 vocabulary size 是 $V$，embedding dimension 是 $d_{\text{model}}$，那么 embedding table 可以看作一个矩阵：
```math
E \in \mathbb{R}^{V \times d_{\text{model}}}
```
其中第 $i$ 行是第 $i$ 个 token 的 embedding vector：

```math
E_i \in \mathbb{R}^{d_{\text{model}}}
```


如果输入 token ID 是 $i$，则 embedding lookup 得到：
```math
\mathbf{x}_i = E_i
```

Embedding 是一个查表函数：
```
E: token ID → R^d_model
```
## 例子

假设：
```math
V = 50000, \quad d_{\text{model}} = 768
```
那么 embedding table 的形状是：
```math
E \in \mathbb{R}^{50000 \times 768}
```
每个 token 会被映射成一个 768 维向量。
# **Intuition**

Embedding 也是模型训练中的一环，它也有一个 weight

Embedding 需要给词表中每一种 token 都准备一个向量

```
Embedding weight: [vocab_size, d_model]
```

⚠️：`vocab_size` 描述的不是输入里有多少 token，而是 tokenizer 一共有多少种 token 可以选择。这个是 Embedding weight 的 shape，不是 input shape


## **Related**

- [Tokenization](<../Language%20modeling/01%20-%20Language%20Modeling%20Basics/Tokenization.md>)
- [Language Modeling](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Language%20Modeling.md>)
- [Transformer](<./Transformer.md>)
- [Neural Network](<../Neural%20Networks/Neural%20Network.md>)