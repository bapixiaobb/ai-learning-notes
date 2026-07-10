#AI #LanguageModeling 

>[!note] Flash Attention 就是 attention 本人，不是 approximate attention
>算的同一个 exact attention，只是换了计算顺序和 memory layout。
>FlashAttention 没有改变 attention 的数学定义，改变的是
>```
>怎么搬数据
>怎么分 tile
>怎么融合 kernel
>backward 时怎么 recompute
>```

# Naive Attention

对于  $$
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
$$
的普通 [[GPU]] 上的实现（⚠️ 这里已经经过 projection 拿到完整的 $Q, K, V$ 了）：
```
1. 读 Q, K
2. 算 score = QK^T
3. 把 score 写回 HBM/ global memory
   
4. 读 score
5. 算 P = softmax(score)
6. 把 P 写回 HBM/ global memory
   
7. 读 P, V
8. 算 O = PV
9. 把 O 写回 HBM/ global memory
```

但是这里 $\mathrm{score} = QK^\top$ 这个 attention score matrix 很大，如果 sequence length 是 $n$，那 $\mathrm{score}$ 的大小是 $n\times n$，所以它是 quadratic memory。而对于 [[GPU Bottleneck]] 其实主要问题还是在于 [[GPU Memory Bound]]，所以这个问题是，它会把巨大的中间矩阵 `score` 和 `P`来回读写，就很浪费 memory bandwidth

---
# FlashAttention

>[!important] 不要 materialize 整个 attention matrix
>不要把完整的 `score` 和 `P` 写到 HBM

Optimization: 
1. 一小块一小块算 `score` 
2. 在 fast memory 里做 $\mathrm{softmax}$
3. 马上 $\times V$
4. 更新输出 `O`
5. 然后丢掉这个 tile

![[FlashAttention.png]]
如上图所示，它不再完整算完整个 tensor，而是 tile by tile 地把 `score = QK^T`、`softmax` 还有 `*V` 融合在一起，这里是一个 [[Operator fusion]] 

---
# Tile [[Softmax]]?

>[!problem] safe softmax 需要知道整行的 $\max(x)$ 和 $\sum e$ 
>所以看起来不能分块

为了配合 FlashAttention 的 tile by tile 计算，**==[[Online Softmax]]==** 


>[!summary] My Understanding
>
>- Tile-wise computation of the inner products, (𝑆)
>- Fusion of the exponential operator 
>- Tile-wise computation of the softmax via the online, telescoping sum trick
>(We won’t cover the backward pass – but they recompute tile-by-tile..)

(这里 backward 的时候，用到了[[Recomputation]]）