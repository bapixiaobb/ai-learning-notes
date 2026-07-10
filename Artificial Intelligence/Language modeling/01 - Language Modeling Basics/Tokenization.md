#LanguageModeling #NLP

**定义**：Tokenization 是把 raw text 切分成 tokens，并映射成 token IDs 的过程。

```math
\text{raw text} \rightarrow \text{tokens} \rightarrow \text{token IDs}
```

## 直觉

Language model 不能直接处理字符串，它只能处理离散 token ID，再通过 [[Embedding]] table 转成向量。

## 常见 token 粒度

| Token 类型 | 切分方式 | 特点 |
|---|---|---|
| **Byte token** | 按 UTF-8 byte 切分 | vocabulary 很小，通常 256；能覆盖任意文本，但 sequence length 较长 |
| **Character token** | 按字符切分 | 比 byte 更接近人类字符直觉，但仍然 sequence length 较长 |
| **Word token** | 按词切分 | sequence length 较短，但 vocabulary 很大，容易遇到 out-of-vocabulary 问题 |
| **Subword token** | 按常见子词片段切分 | 在 vocabulary size 和 sequence length 之间折中，例如 [[Byte Pair Encoding (BPE)]] |

## 影响

Tokenization 会影响：

- vocabulary size
- sequence length
- [[Embedding]] matrix size
- attention cost
- rare word handling
- multilingual text handling

## Related

- [[Byte Pair Encoding (BPE)]]
- [[Embedding]]
- [[Language Modeling]]
- [[Transformer]] 