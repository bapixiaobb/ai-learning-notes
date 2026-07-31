#AI #LanguageModeling #Inference #Transformer

在 [Decoder-Only Transformer](<../../Transformer/Decoder-Only%20Transformer.md>) 的 [autoregressive](<../01%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>) inference 中，如果每生成一个新 token 都重新计算所有 previous tokens 的 [keys 和 values](<../../Transformer/Query%20Key%20Value.md>)，会产生大量重复工作。

KV cache 会缓存 previous tokens 在每一层 attention 中产生的 K 和 V：

```math
K_{\leq t},V_{\leq t}
```

生成下一个 token 时，只需要计算新 token 对应的 Q、K、V，再让新的 query attend to cached keys and values。

>**Important** — 它优化什么
>[Autoregressive Decoding](<../01%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>) 决定 token generation 的依赖关系和选择过程。
>
>KV cache 优化每个 decoding step 的 implementation，避免重复计算 previous tokens 的 K/V。

KV cache：

- 不消除不同 generation steps 之间的 [sequential dependency](<../02%20-%20Training%20and%20Scaling/Training%20vs%20Inference.md#why-training-parallelizes-across-positions-but-generation-does-not>)；
- 用额外 memory 换取更少的重复 compute。

它也是 inference 的主要 memory 成本之一，会随着 sequence length、batch/request 数量、layer 数量和 KV heads 等因素增长。

## Next Questions

- prefill 阶段如何建立 KV cache？
- decode 阶段每一步读取和追加什么？
- KV cache 的 shape 和 memory accounting 是什么？
- MHA、MQA、GQA 为什么具有不同的 KV cache 大小？
- serving 系统如何分配、分页或共享 KV cache？



## Connections

- Inference
- [Autoregressive Decoding](<../01%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>)
- [Multi-Head Attention](<../../Transformer/Multi-Head%20Attention.md>)
- [GPU Memory Bound](<../03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>)
- [Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)
