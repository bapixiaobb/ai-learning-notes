#AI #LanguageModeling

Inference 中的 [Arithmetic Intensity](<../../Fundamentals/Arithmetic%20Intensity.md>) 要分 [Prefill](<Prefill.md>) 和 [Generation](<Generation.md>) 还有 [MLP](<../../Transformer/03%20-%20MLP%20and%20Activations/MLP.md>) 和 [Attention](<../../Transformer/02%20-%20Attention/Self-Attention.md>) 来看

输入为：

```math
X\in\mathbb{R}^{B\times T \times D}
```
## MLP layers

由 [Arithmetic Intensity in MLP](<../../Transformer/03%20-%20MLP%20and%20Activations/Arithmetic%20Intensity%20in%20MLP.md>) 推导出
```math
I = \frac{6BTDF} {4BTD+4BTF+6DF}=\frac{3BTDF}{2BT(D+F)+3DF}
```
在实际 inference 中 $BT\ll D,F$ ，所以 denominator 中 $DF$ 占主导

于是：
```math
I\approx\frac{3BTDF}{3DF}
```
得到：
```math
\boxed{I\approx BT}
```

## Attention layers

在 [Arithmetic Intensity in Attention](<../../Transformer/02%20-%20Attention/Arithmetic%20Intensity%20in%20Attention.md>) 推导出
```math
\boxed{  I = \frac{ST} {S+T}  }
```
在 [Prefill](<Prefill.md>) 阶段 $T=S$
```math
I=S/2
```
在 [Generation](<Generation.md>) 阶段 $T=1$
```math
I=\frac{S}{S+1}<1
```
所以在 generation 阶段很糟糕
