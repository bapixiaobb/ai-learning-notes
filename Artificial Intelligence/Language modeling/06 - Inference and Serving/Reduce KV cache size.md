#AI #LanguageModeling

[Inference](<Inference.md>) 是 [memory bound](<../03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>) 的，其中 [KV Cache](<KV%20Cache.md>) 占大头，所以保证精度的前提下，降低 KV cache 的 size 是一大优化

## [Grouped-Query Attention](<../../Transformer/02%20-%20Attention/Grouped-Query%20Attention.md>)

相比于完整的 [MHA](<../../Transformer/02%20-%20Attention/Multi-Head%20Attention.md>)，GQA 是 queries 共享一组 keys 和 values，直观看出 GQA reduces the KV cache by a factor of N/K

[Ainslie 的论文](<../../Transformer/02%20-%20Attention/Grouped-Query%20Attention.md#inference-%E4%BC%98%E5%8C%96>) 也说明了，GQA improves [Latency](<Latency.md>) 和 [Throughput](<Throughput.md>)

## [Multi-Head Latent Attention](<../../Transformer/02%20-%20Attention/Multi-Head%20Latent%20Attention.md>)

相比于 GQA 的共享 $K$ 和 $V$，MLA 主要是把 $K,V$ 压缩，也能达到减少 KV cache 大小的作用

这个比 GQA 贵一些，但是号称比 GQA 甚至 [MHA](<../../Transformer/02%20-%20Attention/Multi-Head%20Attention.md>) 要好

## [Cross-Layer Attention](<../../Transformer/02%20-%20Attention/Cross-Layer%20Attention.md>)

## [Sliding Window Attention](<../../Transformer/02%20-%20Attention/Sliding%20Window%20Attention.md>)


还有很多别的方法：linear attention / state-space-models (Mamba 2, GatedDeltaNet), diffusion models
