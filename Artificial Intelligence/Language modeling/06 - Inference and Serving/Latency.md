#AI #LanguageModeling

[Inference](<Inference.md>) 中的一个重要指标

> **Note** — 当前 batch 完成一次 decode step、每条 sequence 各产生一个 token，需要多少秒。

每生成一个 token，需要从 HBM 读取：
```
模型 parameters + 当前 batch 的 KV cache
```

所以：
```math
\text{memory} = \text{parameter bytes} + B\times\text{KV-cache bytes per sequence}
```
然后：
```math
\boxed{ \text{latency} = \frac{\text{memory bytes}} {\text{memory bandwidth}} }
```

单位是

```
seconds / decode step
```

🔗
[Throughput](<Throughput.md>)
