#LanguageModeling #Systems #FLOPs

>**Note**
> Dense Transformer 训练计算量常用粗略近似：
>
> ```math
> \text{Training FLOPs} \approx 6ND
> ```
>
> 其中 $N$ 是模型参数量，$D$ 是训练 token 数。

## 符号含义

| 符号 | 含义 |
|---|---|
| $N$ | number of model parameters |
| $D$ | number of training tokens |
| $6ND$ | approximate total training FLOPs |

>**Note**
> 这里的 $D$ 通常不是 document 数量，而是训练过程中处理过的 token 数量。

## 为什么是 $N \times D$

>**Note**
> 对 dense model 来说，每个 token 训练时大部分参数都会参与一次 forward 和 backward 计算。
>
> 因此单个 token 的计算量大约正比于参数量 $N$；训练 $D$ 个 tokens，总计算量就大约正比于 $ND$。

可以粗略理解为：

```math
\text{FLOPs per token}
\propto
N
```

所以：

```math
\text{Total FLOPs}
\propto
N \times D
```

## 为什么系数约为 6

>**Note**
> 对一个 dense neural network，forward pass 中每个参数大约对应一次 multiply-add，通常记为约 2 FLOPs。
>
> Backward pass 通常大约是 forward pass 的 2 倍。
>
> 因此 forward + backward 约为 forward 的 3 倍。

所以：

```math
\text{forward}
\approx
2N
```

```math
\text{forward + backward}
\approx
3 \times 2N
=
6N
```

每个 token 大约需要：

```math
6N
```

FLOPs。

训练 $D$ 个 tokens：

```math
\text{Training FLOPs}
\approx
6ND
```

## 直觉例子

假设：

```math
N = 10^9
```

即模型有 1B parameters，并且：

```math
D = 10^{11}
```

即训练 100B tokens。

那么训练计算量约为：

```math
6ND
=
6 \times 10^9 \times 10^{11}
=
6 \times 10^{20}
```

FLOPs。

## 适用范围

>**Note**
> $6ND$ 是 dense Transformer training 的粗略估算，不是精确公式。

它适合用来做：

- large-scale training compute estimation
- scaling law analysis
- rough training cost comparison
- resource accounting

## Caveats

>**Note**
> $6ND$ 不适用于所有模型和所有设置。它忽略了很多细节，例如 attention 的 sequence length 成本、embedding lookup、optimizer overhead 和 communication overhead。

可能偏离的情况包括：

| 情况 | 原因 |
|---|---|
| [Mixture of Experts (MoE)](<../05%20-%20Architectures%20and%20MoE/Mixture%20of%20Experts%20(MoE).md>) | 每个 token 只激活部分 parameters |
| embedding matrix | token lookup 不会使用整个 embedding matrix |
| long context attention | attention 有 sequence-length-dependent cost |
| distributed training | communication overhead 不在 $6ND$ 中 |
| optimizer | optimizer step 也有额外计算和 memory cost |

## Related

- [FLOPs](<../03%20-%20GPU%20and%20Systems/FLOPs.md>)
- [Resource Accounting](<./Resource%20Accounting.md>)
- [Transformer](<../../Transformer/Transformer.md>)
- [Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>)
- [Scaling Law](<./Scaling%20Law.md>)