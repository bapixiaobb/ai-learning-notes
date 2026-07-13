#AI #MachineLearning #DeepLearning #NLP #LanguageModeling

>**Note**
> **Large Language Model (LLM)** 是大规模的 language model，通常基于 [Transformer](<../../Transformer/Transformer.md>) 架构，通过大量文本数据训练，学习建模 token sequence 的概率分布，并用于文本生成、问答、代码生成、推理等任务。

## 所属位置

>**Note**
> LLM 位于 [Machine Learning](<../../Fundamentals/Machine%20Learning.md>)、[Deep Learning](<../../Neural%20Networks/Deep%20Learning.md>)、[Natural Language Processing](<../../Fundamentals/Natural%20Language%20Processing.md>) 和 [Language Modeling](<./Language%20Modeling.md>) 的交叉处。
>
> 它不是一个单独的机器学习范式，而是现代 language modeling 在大规模数据、参数和算力下形成的模型类别。

可以理解成：

```math
\text{LLM}
\subset
\text{Transformer-based Language Models}
```

## 核心目标

>**Note**
> 大多数 autoregressive LLM 的训练目标是 [Next-token prediction](<../01%20-%20Language%20Modeling%20Basics/Next-token%20prediction.md>)：根据前面的 tokens 预测下一个 token。

```math
p_\theta(x_t \mid x_{<t})
```

其中：

- $x_t$：当前要预测的 token
- $x_{<t}$：前面的上下文 tokens
- $\theta$：模型参数

训练目标通常是最小化 [Cross Entropy Loss](<../01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)。

## 与 Language Modeling 的关系

>**Note**
> [Language Modeling](<./Language%20Modeling.md>) 是任务；LLM 是大规模执行这个任务的模型。
>
> 换句话说，LLM 通常是通过 language modeling objective 训练出来的。

典型 pipeline：

```math
\text{raw text}
\rightarrow
\text{[Tokenization](<../01%20-%20Language%20Modeling%20Basics/Tokenization.md>)}
\rightarrow
\text{token IDs}
\rightarrow
\text{[Embedding](<../../Transformer/Embedding.md>)}
\rightarrow
\text{[Transformer](<../../Transformer/Transformer.md>)}
\rightarrow
\text{logits}
\rightarrow
\text{next-token probability}
```

## 与 Transformer 的关系

>**Note**
> 现代 LLM 大多基于 [Transformer](<../../Transformer/Transformer.md>) 架构，尤其是 decoder-only Transformer，例如 GPT、LLaMA、Qwen 等。

可以理解成：

```math
\text{LLM}
=
\text{Transformer architecture}
+
\text{large-scale language modeling training}
```

其中 Transformer 负责将上下文 tokens 转换成 hidden representations，并输出每个位置的 logits。

## 模型规模

>**Note**
> LLM 中的 “Large” 通常指模型参数量、训练数据量和训练计算量都很大。

常见衡量指标包括：

- number of parameters
- number of training tokens
- training FLOPs
- context length
- inference cost
- memory usage

训练计算量常用近似：

```math
\text{Training FLOPs} \approx 6ND
```

其中：

- $N$：模型参数量
- $D$：训练 token 数

## 常见模型架构技术

### Transformer

>**Note**
> [Transformer](<../../Transformer/Transformer.md>) 是现代 LLM 的基础架构。它通过 [Self-Attention](<../../Transformer/Self-Attention.md>) 建模 token 之间的上下文关系。

### Mixture of Experts

>**Note**
> [Mixture of Experts (MoE)](<../05%20-%20Architectures%20and%20MoE/Mixture%20of%20Experts%20(MoE).md>) 是一种常见的 LLM 架构扩展。MoE Transformer 会包含多个 experts，但每个 token 通常只激活其中一部分 experts。

MoE 的核心思想是：

```math
\text{large total parameters}
\quad
\text{but fewer active parameters per token}
```

>**Note**
> 因此 MoE 可以增加模型总参数量，同时控制每个 token 的计算成本。

## 训练与推理

>**Note**
> LLM 的训练阶段主要是在大规模文本上学习参数；推理阶段则是根据 prompt 逐步生成 tokens。

训练：

```math
\min_\theta \mathcal{L}(\theta)
```

推理：

```math
x_1, x_2, \dots, x_t
\rightarrow
x_{t+1}
```

>**Note**
> 训练时通常可以并行处理一整段 sequence；而 autoregressive inference 通常需要逐 token 生成。

## 与 Foundation Model 的关系

>**Note**
> LLM 通常也是一种 [Foundation Model](<./Foundation%20Model.md>)。它先通过大规模预训练获得通用语言能力，再通过 instruction tuning、alignment 或 fine-tuning 适配具体任务。

常见阶段包括：

- pretraining
- supervised fine-tuning
- instruction tuning
- alignment
- inference / deployment

## Training Systems

>**Note**
> LLM 训练需要大量 compute、memory 和 hardware resources，因此训练前通常需要做 [Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)，估算 FLOPs、显存、带宽、训练时间和硬件利用率。

相关内容：

- [Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)
- [Training Compute - 6ND](<../02%20-%20Training%20and%20Scaling/Training%20Compute%20-%206ND.md>)
- [Model FLOPs Utilization](<../03%20-%20GPU%20and%20Systems/Model%20FLOPs%20Utilization.md>)
## Related

- [Language Modeling](<./Language%20Modeling.md>)
- [Natural Language Processing](<../../Fundamentals/Natural%20Language%20Processing.md>)
- [Deep Learning](<../../Neural%20Networks/Deep%20Learning.md>)
- [Transformer](<../../Transformer/Transformer.md>)
- [Self-Attention](<../../Transformer/Self-Attention.md>)
- [Tokenization](<../01%20-%20Language%20Modeling%20Basics/Tokenization.md>)
- [Embedding](<../../Transformer/Embedding.md>)
- [Cross Entropy Loss](<../01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)
- [Training Compute - 6ND](<../02%20-%20Training%20and%20Scaling/Training%20Compute%20-%206ND.md>)
- [Mixture of Experts (MoE)](<../05%20-%20Architectures%20and%20MoE/Mixture%20of%20Experts%20(MoE).md>)
- [Foundation Model](<./Foundation%20Model.md>)