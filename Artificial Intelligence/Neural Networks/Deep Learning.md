#AI
[[Artificial Intelligence (AI)]] > [[Machine Learning]] 

**定义**：深度学习是 [[Machine Learning]] 的一个子集，主要研究使用多层 [[Neural Network]] 从数据中学习复杂表示和模式。

与传统机器学习相比，深度学习通常减少对人工特征工程的依赖，能够通过多层结构自动学习从低级到高级的特征表示。

## 核心形式

深度学习模型通常可以看作复合函数：

```math
f_\theta(x)
=
f_L \circ f_{L-1} \circ \cdots \circ f_1(x)
```

其中 $\theta$ 表示所有可学习参数。

## 核心特点

- **深层架构**：包含多个 hidden layers，可以表达复杂 nonlinear mappings。
- **自动特征提取**：从 raw data 中学习 representation。
- **端到端训练**：通常通过 [[Backpropagation]] 和 [[Gradient Descent]] 优化参数。
- **大规模数据与计算**：模型性能通常依赖大量数据和算力。

## 常见模型结构

- **[[Convolutional Neural Network (CNN)]]**：常用于图像和空间结构数据。
- **[[Recurrent Neural Network (RNN)]]**：常用于序列数据。
- **[[Transformer]]**：现代 [[Natural Language Processing]] 和 [[Language Modeling]] 的核心结构。
- **[[Generative Adversarial Network (GAN)]]**：常用于生成任务。
- **Deep Reinforcement Learning**：将 [[Deep Learning]] 和 [[Reinforcement Learning]] 结合。

## 常见应用方向

- [[Natural Language Processing]]
- Computer Vision
- Recommender Systems
- [[Foundation Model]]
- [[Large Language Model (LLM)]]

## Systems Aspect

>[!note]
> 大规模 deep learning 不只涉及模型结构和优化算法，还涉及训练系统效率，例如 compute、memory、communication 和 hardware utilization。
>
> 相关内容见 [[Deep Learning Systems]] 和 [[Resource Accounting]]。
## Related

- [[Machine Learning]]
- [[Neural Network]]
- [[Backpropagation]]
- [[Gradient Descent]]
- [[Transformer]]
- [[Language Modeling]]