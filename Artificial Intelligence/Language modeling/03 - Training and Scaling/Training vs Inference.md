#DeepLearning #NeuralNetwork #Transformer #LanguageModeling #Training #Inference

Training vs Inference 描述的是同一个模型在两个不同运行阶段的行为差异：

- **Training**：用数据和 loss 学习参数。
- **Inference**：固定已经训练好的参数，根据输入计算预测或生成输出。

>**Important** — 概念边界
>Inference 是一个运行阶段；[Autoregressive Decoding](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>) 是 decoder-only language model 在 inference 阶段生成文本的一种算法。
>
>因此二者不是同义词：
>
>```math
>\text{Autoregressive Decoding}
>\subset
>\text{Inference}
>```
>
>Lecture 10 中的 prefill、[KV Cache](<../06%20-%20Inference%20and%20Serving/KV%20Cache.md>)、latency、throughput、batching 和 serving 属于 Inference 的执行与系统层。

## Core Difference

| Aspect | Training | Inference |
|---|---|---|
| parameters | updated | fixed |
| input | training data | prompt / request |
| forward pass | yes | yes |
| target labels | yes | ordinary generation 中没有 |
| loss | used as the optimization objective | ordinary generation 中通常不需要 |
| backward pass | yes | no |
| optimizer step | yes | no |
| output use | compute loss and gradients | prediction or generated tokens |
| dropout | usually on if the recipe uses it | usually off |
| KV cache | usually not the main execution mode | commonly used for autoregressive generation |

>**Note**
>Inference 不是“模型开始工作”，training 也不是“没有 forward pass”。
>
>两者都会执行 model forward；training 还会计算 loss、执行 [backward pass](<../../Neural%20Networks/Backpropagation.md>)，再由 [Optimizer](<../../Transformer/05%20-%20Training/Optimizer.md>) 更新参数。

## Execution Flow

### Training

```text
training tokens
→ model forward
→ logits at many positions
→ loss
→ backward
→ gradients
→ optimizer step
→ updated parameters
```

对于 decoder-only language model，训练数据的 input 和 target 通常是同一 token sequence 的 shifted versions：

```math
[x_1,x_2,x_3,x_4]
\rightarrow
[x_2,x_3,x_4,x_5]
```

causal mask 保证每个位置只能使用 prefix，但由于整段正确 token 已经存在，Transformer 可以一次并行计算多个 positions 的 logits 和 loss。

更详细的训练目标见 [Next-token prediction](<../02%20-%20Language%20Modeling%20Basics/Next-token%20prediction.md>)、[Cross Entropy Loss](<../02%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>) 和 [Training Recipe](<Training%20Recipe.md>)。

### Inference

```text
input / prompt
→ model forward
→ prediction or logits
→ task-specific output processing
```

Inference 本身不等于文本生成。例如，分类模型做一次 forward 得到类别预测也是 inference。

对于 decoder-only language model 的文本生成，常见 output processing 是 [Autoregressive Decoding](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>)：

```text
last-position logits
→ probability distribution
→ choose/sample next token
→ append token
→ repeat
```

temperature、top-k 和 top-p 等 token-selection 细节统一放在 [Autoregressive Decoding](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>)，不在这里重复。

## Why Training Parallelizes Across Positions but Generation Does Not

**训练**时，sequence 中所有正确 tokens 已经给定：

```math
[x_1,x_2,\dots,x_T]
\rightarrow
[z_1,z_2,\dots,z_T]
```

causal mask 只限制 information dependency，不妨碍这些 positions 在一次 forward 中并行计算。

**autoregressive generation** 时，未来 token 尚未产生：

```math
x_{T+1}
\rightarrow
x_{T+2}
\rightarrow
x_{T+3}
```

必须先选出 $x_{T+1}$，才能把它作为 context 生成 $x_{T+2}$。因此不同 generation steps 之间存在 [sequential dependency](<../06%20-%20Inference%20and%20Serving/Generation.md#sequential>)。

## Execution Mode

一些 modules 在 training 和 inference 中行为不同，例如 Dropout：

- `model.train()`：启用 training mode；
- `model.eval()`：启用 evaluation/inference mode。

这两个调用不会直接更新参数，只会改变 dropout、batch normalization 等特定 modules 的行为。普通 inference 通常还会关闭 gradient recording。

## Compute and Memory

Training 需要保存或维护：

- backward 所需的 activations；
- gradients；
- optimizer states；
- parameter updates。

Inference 不需要普通训练中的 backward 和 optimizer states，但有自己的系统成本：

- model weights；
- [KV Cache](<../06%20-%20Inference%20and%20Serving/KV%20Cache.md>)；
- autoregressive decode latency；
- concurrent requests；
- long-context attention；
- serving throughput。

| Main concern | Training | Inference |
|---|---|---|
| optimization | loss and parameter updates | fixed-parameter execution |
| parallelism | dense tokens and large training batches | requests, prefill and decode scheduling |
| memory | parameters, activations, gradients, optimizer states | parameters and KV cache |
| performance | training throughput / time-to-train | latency, throughput and memory capacity |

这些 inference-specific 问题统一从 Inference 继续展开。

## Common Confusions

### Inference 不是没有 forward pass

Inference 的核心仍然是 forward pass，只是不执行普通训练中的 backward 和 optimizer step。

### Training 不是一次只预测一个 token

decoder-only LM 训练时可以并行计算多个 positions 的 next-token predictions。

### Autoregressive decoding 不等于 inference

它是语言模型在 inference 阶段生成 sequence 的一种方式。Inference 还包括其他模型任务，以及语言模型的执行、优化和 serving。

### KV cache 不是 decoding strategy

temperature、top-k、top-p 决定如何选择 token；[KV Cache](<../06%20-%20Inference%20and%20Serving/KV%20Cache.md>) 决定如何避免重复计算 previous tokens 的 K/V。前者属于生成算法，后者属于 inference implementation。

### Decoding strategy 不等于 model architecture

改变 temperature 或 top-p 不会改变 Transformer architecture；它改变的是如何从模型给出的 distribution 中选择 token。

## My Understanding

>**Summary**
>Training 用 data、loss、backward 和 optimizer 学习参数；inference 固定参数并使用 model forward 得到预测或生成输出。
>
>对于 decoder-only language model，[Autoregressive Decoding](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>) 描述“如何逐 token 生成”，而 Inference 还要研究“这套生成过程如何高效执行和服务”。

## Connections

- [Forward Propagation](<../../Neural%20Networks/Forward%20Propagation.md>)
- [Backpropagation](<../../Neural%20Networks/Backpropagation.md>)
- [Optimizer](<../../Transformer/05%20-%20Training/Optimizer.md>)
- [Model Architecture](<../02%20-%20Language%20Modeling%20Basics/Model%20Architecture.md>)
- [Training Recipe](<Training%20Recipe.md>)
- [Next-token prediction](<../02%20-%20Language%20Modeling%20Basics/Next-token%20prediction.md>)
- [Cross Entropy Loss](<../02%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)
- [Autoregressive Decoding](<../02%20-%20Language%20Modeling%20Basics/Autoregressive%20Decoding.md>)
- Inference
- [KV Cache](<../06%20-%20Inference%20and%20Serving/KV%20Cache.md>)
- [Systems for Language Models](<../01%20-%20System/Systems%20for%20Language%20Models.md>)
- [Resource Accounting](<Resource%20Accounting.md>)
