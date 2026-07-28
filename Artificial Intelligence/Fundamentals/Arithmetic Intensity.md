#AI #LanguageModeling #LLM


how much compute we do per byte transferred (want to be high)

```math
I_{\text{arith}}
=
\frac{\text{FLOPs}}{\text{bytes moved}}
```

## Example

```
multiply X (B x D) and W (D x F)

B: batch size
D: hidden dimension
F: up-projection dimension in MLP
```

Perform the operation
```
1. Read X from HBM: bytes_transferred += 2*B*D
2. Read W from HBM: bytes_transferred += 2*D*F
3. Compute Y = X @ W: flops += 2*B*D*F
4. Write Y to HBM: bytes_transferred += 2*B*F

flops = 2*B*D*F
bytes_transferred = 2*B*D + 2*D*F + 2*B*F

intensity = B*D*F / (B*D + D*F + B*F)
```