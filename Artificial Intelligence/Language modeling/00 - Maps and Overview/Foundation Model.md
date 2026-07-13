#AI #MachineLearning #DeepLearning

>**Note**
> **Foundation Model** 是一种在大规模数据上预训练得到的通用模型。它可以作为许多下游任务的基础，再通过 prompting、fine-tuning 或 instruction tuning 适配具体任务。

## 核心直觉

>**Note**
> Foundation model 的核心思想是：先用大规模数据训练一个通用表示模型，再把这个模型迁移到不同任务中使用。
>
> 它不是为某一个单一任务从零训练的模型，而是一个可以被复用、改造和扩展的基础模型。

## 与传统模型的区别

| 类型 | 训练方式 | 使用方式 |
|---|---|---|
| 传统任务模型 | 针对单一任务训练 | 只解决特定任务 |
| Foundation Model | 大规模预训练 | 可迁移到多种任务 |

>**Note**
> 例如，传统 sentiment classification model 只做情感分类；而一个 foundation model 可以通过不同 prompt 或 fine-tuning 做情感分类、问答、摘要、翻译、代码生成等任务。

## 常见例子

- [Large Language Model (LLM)](<./Large%20Language%20Model%20(LLM).md>)
- Vision foundation models
- Multimodal foundation models
- Speech foundation models

>**Note**
> 在 NLP 中，现代 [Large Language Model (LLM)](<./Large%20Language%20Model%20(LLM).md>) 通常就是一种 foundation model，例如 GPT、LLaMA、Qwen 等。

## 训练流程

Foundation model 通常包含几个阶段：

1. **Pretraining**
2. **Fine-tuning**
3. **Instruction Tuning**
4. **Alignment**
5. **Inference / Deployment**

## Pretraining

>**Note**
> Pretraining 是 foundation model 的基础阶段。模型在大规模数据上学习通用模式，而不是只学习某个具体任务。

对于 language model，常见预训练目标是：

```math
p_\theta(x_t \mid x_{<t})
```

也就是根据前面的 tokens 预测下一个 token。

## Adaptation

>**Note**
> 预训练后的 foundation model 可以通过不同方式适配下游任务。

常见方法包括：

- **Prompting**：直接用 prompt 引导模型完成任务。
- **Fine-tuning**：在特定任务数据上继续训练模型参数。
- **Instruction Tuning**：用指令-回答数据让模型更会遵循自然语言指令。
- **Alignment**：让模型输出更符合人类偏好、安全性或任务要求。

## 与 LLM 的关系

>**Note**
> [Large Language Model (LLM)](<./Large%20Language%20Model%20(LLM).md>) 是 foundation model 在自然语言领域的典型形式。
>
> 但 foundation model 不只包括 LLM，也可以包括 vision、speech、multimodal 等领域的大规模预训练模型。

可以理解成：

```math
\text{LLM}
\subset
\text{Foundation Model}
```

但不是所有 foundation model 都是 LLM。

## 与 Language Modeling 的关系

>**Note**
> 在 NLP 中，许多 foundation models 是通过 [Language Modeling](<./Language%20Modeling.md>) 任务预训练出来的。
>
> Language modeling 提供训练目标，foundation model 是训练后得到的通用模型。

例如 autoregressive LLM 的目标是：

```math
\min_\theta \mathcal{L}(\theta)
```

其中 loss 通常来自 [Next-token prediction](<../01%20-%20Language%20Modeling%20Basics/Next-token%20prediction.md>) 的 [Cross Entropy Loss](<../01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)。

## 为什么重要

>**Note**
> Foundation model 改变了机器学习的工作流：从“为每个任务单独训练模型”，转向“预训练一个通用模型，再适配多个任务”。

它的重要性主要来自：

- 可以复用大规模预训练能力
- 支持多任务迁移
- 能通过 prompt 完成新任务
- 是现代 LLM 和 multimodal AI 的基础
- 对数据、算力和系统设计提出更高要求

## Related

- [Large Language Model (LLM)](<./Large%20Language%20Model%20(LLM).md>)
- [Language Modeling](<./Language%20Modeling.md>)
- [Transformer](<../../Transformer/Transformer.md>)
- Pretraining
- Fine-tuning
- Instruction Tuning
- Alignment
- [Cross Entropy Loss](<../01%20-%20Language%20Modeling%20Basics/Cross%20Entropy%20Loss.md>)
- [Scaling Law](<../02%20-%20Training%20and%20Scaling/Scaling%20Law.md>)