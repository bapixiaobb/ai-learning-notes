#AI #LanguageModeling

一种 [Inference](<Inference.md>) 优化方法

Key idea: just rip out parts of an expensive model to make it cheaper
...and then fix it up.

Paper from NVIDIA  [Muralidharan+ 2024](https://arxiv.org/abs/2407.14679)

![pruning-kd-loop](<../../attachments/pruning-kd-loop.png>)

Algorithm:

1. Identify important {layer, head, hidden dimension} on a small calibration dataset (1024 samples)
2. Remove unimportant layers to get a smaller model
3. Distill the original model into pruned model
