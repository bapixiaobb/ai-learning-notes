#AI #LanguageModeling

这里我们主要讲 Causal Multi-Head [Self-Attention](<./Self-Attention.md>)，这里的 Causal 指的是 [Causal Mask](<./Causal%20Mask.md>)

Recall that, mathematically, the operation of applying multi-head attention is defined as follows:
```math
\text{MultiHead(Q, K, V)} = \text{Concat}(\text{head}_1,\cdots,\text{head}_h)
```
```math
\text{for head}_i=\text{Attention}(Q_i,K_i,V_i)
```
with $𝑄_𝑖$, $𝐾_𝑖$, $𝑉_i$  being slice number $𝑖\in\{1,...,h\}$ of size $𝑑_𝑘$ or $𝑑_𝑣$ of the embedding dimension for [Query Key Value](<./Query%20Key%20Value.md>) respectively.

From this we can form the multi-head self-attention operation:
```math
\text{MultiHeadSelfAttention}(𝑥) = 𝑊_𝑂 \text{MultiHead}(𝑊_𝑄𝑥, 𝑊_𝐾 𝑥, 𝑊_𝑉 𝑥)
```

- $W_Q\in\mathbb{R}^{hd_k\times d_{\text{model}}}$
- $W_K\in\mathbb{R}^{hd_k\times d_{\text{model}}}$
- $W_V\in\mathbb{R}^{hd_v\times d_{\text{model}}}$
- $W_O\in\mathbb{R}^{d_{\text{model}}\times hd_v}$

---
```
h = num_heads
d_model = 每个 token 的完整向量长度
d_k = 每个 head 里 query/key 的长度
d_v = 每个 head 里 value 的长度
```

一般来说，常见设置是：
```
d_k = d_v = d_model / h   -->   d_model = h * d_k = h * d_v

d_head = d_k = d_v
```
---
## Shape derivation

Input of Attention $x$ (a group of token vectors):
```
x = [..., sequence_length, d_model]
```
Projection:
```
q_proj_weight = [d_model, d_model]
```
Full `Q` shape is:
```
Q = [..., sequence_length, d_model]
```
Split heads:
```
d_model = num_heads * d_head --> Q = [..., sequence_length, num_heads, d_head]
```
Move head dimension before sequence dimension:
```
Q = [..., num_heads, sequence_length, d_k]
```
Attention per head:
```
Q_i = [..., sequence_length, d_k]
K_i = [..., sequence_length, d_k]
V_i = [..., sequence_length, d_v]
```
Next is scaled dot-product attention:
```
Softmax(Q @ K.T / sqrt(d_k)) @ V
```
Score:
```
score = Q @ K.T = [..., num_heads, sequence_length, sequence_length]
```
Attention weights:
```
weight = Softmax(score / sqrt(d_k)) = [..., num_heads, sequence_length, sequence_length]
```
Output:
```
Output =  weight @ V = [..., num_heads, sequence_length, d_v]
```
Combination:
```
concat_heads = [..., sequence_length, d_model]
```
Output projection:
```
o_proj_weight = [d_model, d_model]
```
Final:
```
out = concat_heads @ o_proj_weight.T = [..., sequence_length, d_model]
```

