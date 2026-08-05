#AI #LanguageModeling

这主要是为了 [Reduce KV cache size](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Reduce%20KV%20cache%20size.md>) 用的一种方法

![gmqa](<../../attachments/gmqa.png>)
👈 是 [Multi-Head Attention](<Multi-Head%20Attention.md>) 有完整的 $H$ 个 [query, key and value heads](<Query%20Key%20Value.md>)
👉 是 Multi-Query Attention 有 $H$ 个 query heads 但是只有 1 个 key 和 value head
👆是 Grouped-Query Attention 有 $N$ query heads, 但是 $K$ key 和 value heads (interacting with N/K query heads)

- Multi-headed attention (MHA): K=N
- Multi-query attention (MQA): K=1
- Group-query attention (GQA): K is somewhere in between

## [Inference](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Inference.md>) 优化
[Latency](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Latency.md>)/[Throughput](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Throughput.md>) improves:  [Ainslie+ 2023](https://arxiv.org/pdf/2305.13245.pdf)
![gqa-speed](<../../attachments/gqa-speed.png>)

Why does GQA improve latency and throughput?
GQA reduces the KV cache by a factor of N/K.

> **Important** — 论文里同样验证了，accuracy 没有降 （对特定模型，有些模型会有影响）
