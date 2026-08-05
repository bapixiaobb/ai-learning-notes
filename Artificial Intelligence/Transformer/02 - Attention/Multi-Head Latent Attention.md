#AI #LanguageModeling

这主要是 [DeepSeek-AI+ 2024](https://arxiv.org/abs/2405.04434) 用来 [Reduce KV cache size](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Reduce%20KV%20cache%20size.md>) 的一种方法

![mla-schema](<../../attachments/mla-schema.png>)

MLA 主要是对 [Query Key Value](<Query%20Key%20Value.md>) 的一种压缩的想法💡

原始的 [Multi-Head Attention](<Multi-Head%20Attention.md>) 来说，[KV Cache](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/KV%20Cache.md>) 就是 $K = W_K H$, $V = W_V H$ ($N\times H$ dimensions)

MLA 的想法是，储存 compressed vector $C=W_C h$ ($c$ dimension)，当有需要的时候 project up

```math
K=W_K C
```
```math
V=W_V C
```

DeepSeek v2: reduce $N*H = 16384$ to $C = 512$

⚠️ MLA 和 original [RoPE](<../01%20-%20Inputs%20and%20Position/Rotary%20Position%20Embedding.md>) 不协同，所以用 MLA 的时候要给 RoPE 加另外的 $64$ dimensions （decoupled RoPE）

#### [Inference](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Inference.md>) 优化
这个方法和 [GQA](<Grouped-Query%20Attention.md>) 一样，减少了 [memory](<../../Language%20modeling/03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>) -> [Latency](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Latency.md>)/[Throughput](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Throughput.md>) improvement

论文里论证了
- MHA is better than GQA (though more expensive)
- MLA is even a bit better than [MHA](<Multi-Head%20Attention.md>) (真的吗😟)
