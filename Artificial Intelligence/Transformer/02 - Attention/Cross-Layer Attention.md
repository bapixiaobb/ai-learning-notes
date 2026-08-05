#AI #LanguageModeling

一种 [Reduce KV cache size](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Reduce%20KV%20cache%20size.md>) 的方法 [Brandon+ 2024](https://arxiv.org/abs/2405.12981)
![cla-diagram](<../../attachments/cla-diagram.png>)

## 💡 Idea

就想 [GQA](<Grouped-Query%20Attention.md>) 共享 [KVs](<Query%20Key%20Value.md>) across heads 一样，CLA 是共享 KVs across **layers**

Empirically improves the pareto frontier of accuracy and KV cache size ([Latency](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Latency.md>) and [Throughput](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Throughput.md>))
