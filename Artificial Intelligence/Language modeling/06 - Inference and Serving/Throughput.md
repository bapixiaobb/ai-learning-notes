#AI #LanguageModeling

[Inference](<Inference.md>) 中的一个重要指标

假设 [Batch Size](<../02%20-%20Language%20Modeling%20Basics/Batch%20Size.md>) 是 $B$，一次 decode step 会产生 $B$ 个 tokens，所以：
```math
\boxed{ \text{throughput} = \frac{B}{\text{latency}}}
```
单位是

```
tokens / second
```

🔗
[Latency](<Latency.md>)
