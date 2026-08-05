#AI #LanguageModeling

因为在 [MLP](<MLP.md>) 具体运算中 Matmul 占大头，所以我们在计算 [Arithmetic Intensity](<../../Fundamentals/Arithmetic%20Intensity.md>) 时只考虑 Matmul

这里直接考虑 [SwiGLU](<SwiGLU.md>)

输入可以写成：
```math
X\in\mathbb{R}^{B\times S \times D}
```

```
1. Read X from HBM
   bytes_transferred += 2*B*S*D
2. Read W_1 W_2 W_3 from HBM
   bytes_transferred += 3 * 2*D*F
3. Compute Upper projection U = X @ W_1
   flops += 2*B*S*D*F
4. Write U to HBM
   bytes_transferred += 2*B*S*F
5. Compute gate G = X @ W_2
   flops += 2*B*S*D*F
6. Write G to HBM
   bytes_transferred += 2*B*S*F
7. Compute Y = SiLU(G) * U @ W_3
   flops += 2*B*S*D*F
8. Write Y (B x S x D) to HBM
   bytes_transferred += 2*B*S*D

flops = 6*B*S*D*F
bytes_transferred == 4*B*S*D + 4*B*S*F + 6*D*F

intensity = 6*B*S*D*F / (4*B*S*D + 4*B*S*F + 6*D*F)
```

所以这里
```math
I = \frac{6BSDF} {6DF+4BS(D+F)}
```
消掉 2：
```math
\boxed{  I = \frac{3BSDF} {3DF+2BS(D+F)}  }
```
