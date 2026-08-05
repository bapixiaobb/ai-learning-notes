#AI #LanguageModeling

[Inference](<Inference.md>) 的一种很神奇的优化方法

inference 中的两步：
- [Prefill](<Prefill.md>): compute-bound (note: also gives you probabilities)
- Generation: memory-bound
简单来说就是：检查比生成快

## 💡 Idea
Speculative sampling  [Leviathan+ 2022](https://arxiv.org/abs/2211.17192)[Chen+ 2023](https://arxiv.org/abs/2302.01318)
- Use a cheaper **draft model** p to guess a few tokens (e.g., 4)
- Evaluate with target model q (process tokens in parallel), and accept if it looks good
[Speculative sampling video](https://storage.googleapis.com/gweb-research2023-media/media/SpeculativeDecoding-1-Illustration.mp4) [article](https://research.google/blog/looking-back-at-speculative-decoding/)

> 小模型 (draft model) 先便宜地 autoregressive decode 一小段，大模型 (target model) 再一次性检查整段。

## pseudocode
![speculative-sampling-algorithm](<../../attachments/speculative-sampling-algorithm.png>)

This is modified rejection sampling with proposal p and target q
Modification: always generate at least one candidate (rejection sampling will keep looping)
Key property: guaranteed to be an **exact sample** from the target model!

## Proof by example:
```
assume two vocabulary elements {A, B}
- Target model probabilities: [q(A), q(B)]
- Draft model probabilities: [p(A), p(B)]
- Assume p(A) > q(A) [draft model oversamples A].
- Therefore p(B) < q(B) [draft model undersamples B].
- Residual probabilities max(q-p, 0): [0, 1]
Compute the probabilities of speculatively sampling a token:
- P[sampling A] = p(A) * (q(A) / p(A)) + p(B) * 1 * 0 = q(A)
- P[sampling B] = p(B) * 1 + p(A) * (1 - q(A) / p(A)) * 1 = q(B)
```

In practice:
- Target model has 70B parameters, draft model has 8B parameters
- Target model has 8B parameters, draft model has 1B parameters
- Try to make draft model as close to target (distillation)
