在显存放不下 large batch 时，用多个 microbatches 累加梯度，再做一次 optimizer step。

---
基本思想是：

1. 用多个 smaller microbatches 分别计算 gradients；
2. 不立即更新参数；
3. 把 gradients 累积起来；
4. 累积到一定步数后再做一次 optimizer step。

形式上，如果累积 $K$ 个 microbatches：

$$
g
=
\frac{1}{K}
\sum_{k=1}^{K}
g_k
$$

然后再用 $g$ 更新参数。

有效 batch size：

$$
B_{\text{effective}}
=
B_{\text{micro}}
\times
K_{\text{accum}}
\times
N_{\text{devices}}
$$

对于 language model，常写成 global batch tokens：

$$
\text{global batch tokens}
=
B_{\text{micro}}
\times
S
\times
K_{\text{accum}}
\times
N_{\text{devices}}
$$

其中 $$S$$ 是 sequence length。

>[!note]
> Gradient accumulation 不减少总 FLOPs；它主要是在有限显存下实现更大的 effective batch size。