#AI #LanguageModeling

[Inference](<Inference.md>) 开头的动作

把用户已经给出的整段 prompt 一次性送进 [Transformer](<../../Transformer/00%20-%20Maps%20and%20Architectures/Transformer.md>):
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
## Prefill 是 [compute-bound](<../../GPU%20and%20NPU/GPU%20Bottleneck.md>)

要分两层来看

#### MLP intensity


Prefill MLP intensity: `B*S`

Prefill attention intensity: `S/2`
