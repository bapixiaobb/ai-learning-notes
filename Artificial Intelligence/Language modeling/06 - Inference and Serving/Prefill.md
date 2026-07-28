#AI #LanguageModeling

Inference 开头的动作

把用户已经给出的整段 prompt 一次性送进 [Transformer](<../../Transformer/Transformer.md>):
```
input
[batch, seq]
    ↓
TransformerLM
    ↓
logits
[batch, seq, vocab_size]
```

在 inference 系统的术语中：

```
整段 prompt [x₁,x₂,x₃,x₄]
第一次完整通过 Transformer
```

这一次 forward 就叫做 **prefill**。