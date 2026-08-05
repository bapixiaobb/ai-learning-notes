#AI #LanguageModeling

和 [MLP 的 arithmetic intensity 计算](<../03%20-%20MLP%20and%20Activations/Arithmetic%20Intensity%20in%20MLP.md>) 相同，这里也只考虑 [Decoder-Only Transformer](<../00%20-%20Maps%20and%20Architectures/Decoder-Only%20Transformer.md>) 中的 [self](<Self-Attention.md>) [causal attention](<Causal%20Attention.md>) 中的 Matmul 计算

输入写成：
```math
X\in\mathbb{R}^{B\times T \times D}
```
Inference 中的 [Query Key Value](<Query%20Key%20Value.md>) 和 training 里的 shape 会不一样，training 中完整的 shape 推导见 [Implementation of Multi-head Attention](<Implementation%20of%20Multi-head%20Attention.md>)，这里我们用更 general 的场景，也就是

```math
Q\in\mathbb{R}^{T\times d}
```

```math
K,V\in\mathbb{R}^{S\times d}
```

这里：

- `T`：有多少个 query 或者说输入的 sequence length，也就是现在要给多少个 token positions 计算输出；
- `S`：有多少个 key/value，也就是每个 query 可以查看多少个 context positions，看多少上下文。

在 **training** 中如果是普通 full [self-attention](<Self-Attention.md>) 那么
```math
\boxed{T=S=\text{seq}}
```
```
1. Read Q , K , V from HBM
   bytes_transferred += 2*B*T*D + 2*B*S*D + 2*B*S*D
2. Compute A = Q @ K^T
   flops += 2*B*S*T*D
3. Compute Y = softmax(A) @ V 
   flops += 2*B*S*T*D
4. Write Y to HBM
   bytes_transferred += 2*B*T*D

flops == 4*B*S*T*D
bytes_transferred == 4*B*S*D + 4*B*T*D

intensity == 4*B*S*T*D / (4*B*S*D + 4*B*T*D)
```

```math
\boxed{  I = \frac{ST} {S+T}  }
```
