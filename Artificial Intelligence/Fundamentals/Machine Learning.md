#AI 

**定义**：机器学习是 **[[Artificial Intelligence (AI)]] 的一个子集**，主要研究如何让计算机**从数据中学习模式**，并**基于这些模式进行预测或决策**。

机器学习的核心形式通常可以写成：

```math
\min_\theta \mathcal{L}(\theta)
```

其中模型从数据中学习参数 $\theta$，使 loss function 尽可能小。

## 机器学习主要类型

| **机器学习类型** | 特点 | 常用方法 |
| :--- | :---: | ---: |
| **监督学习（[[Supervised Learning]]）** | 有标签数据，目标是预测或分类 | 线性回归、逻辑回归、决策树、SVM、神经网络 |
| **无监督学习（[[Unsupervised Learning]]）** | 无标签数据，目标是发现结构或模式 | K-means、PCA、密度估计 |
| **半监督学习（[[Semi-supervised Learning]]）** | 部分数据有标签，部分无标签 | 伪标签、一致性正则化 |
| **强化学习（[[Reinforcement Learning]]）** | 智能体通过和环境交互，根据 reward 学习策略 | Q-learning、DQN、Policy Gradient |
| **深度学习（[[Deep Learning]]）** | 使用多层神经网络学习复杂表示 | CNN、RNN、Transformer |
## 常见应用方向

- [[Natural Language Processing]]
- Computer Vision
- Speech Recognition
- Recommender Systems
- Robotics
## Related
- [[Artificial Intelligence (AI)]]
- [[Deep Learning]]
- [[Neural Network]]
- [[An Optimization Problem]]
- [[Statistical Learning]]