#AI #Transformer

Mixture of Experts，简称 MoE，是一种在 neural network 中引入多个 experts，并根据输入动态选择部分 experts 参与计算的 architecture。

在 modern language model 中，MoE 最常见的形式是：
用 MoE layer 替换 Transformer block 里的 [Feed-Forward Network](<../../Neural%20Networks/Feed-Forward%20Network.md>) / [MLP](<../../Transformer/MLP.md>)。

---
## 🧠 Core Idea

>**Note**
>MoE 的核心思想是 **conditional computation**。
>
>Dense model 中，每个 token 通常都会经过同一组参数。
>
>MoE 中，每个 token 只激活一部分 experts，因此可以增加 total parameters，但不同比例地增加每个 token 的 compute cost。

可以粗略理解为：
```math
\text{Dense FFN}
\rightarrow
\text{many experts}
\rightarrow
\text{route each token to a few experts}
```

也就是说 MoE 试图做到：
```math
\text{more total parameters}
\quad
\text{without proportional increase in active computation}
```
---
## 🧩 MoE in Transformer

在 standard Transformer block 中，通常有：
```math
\text{Attention}
\rightarrow
\text{MLP / FFN}
```
MoE 的常见做法是把 FFN 替换成多个 experts：
```math
\text{Attention}
\rightarrow
\text{MoE Layer}
```
其中每个 expert 通常本身就是一个 FFN：
```math
\text{Expert}_i(x)=
\mathrm{FFN}_i(x)
```
所以 MoE 在 LLM 中经常可以理解成
>**Note**
>MoE 是一种更稀疏、更可扩展的 FFN/MLP 替代方案。

---
## 🧱 Two Main Parts
MoE主要由两个部分组成

|**Part**|**Role**|
|---|---|
|experts|真正处理 token 的 neural networks，通常是 FFNs|
|router / gate|决定每个 token 应该送到哪些 experts|
#### Experts

每个 expert 是一个独立的 neural network。
在 Transformer MoE 中，expert 通常是 FFN / MLP。
如果有 $N$ 个 experts：$ E_1, E_2, ..., E_N$
每个 token 不会经过所有 experts，而是指选择其中一小部分

#### Router / Gate

Router 负责为每个 token 计算 experts scores，并选择 top-k experts。
对于 token representation $x$，router 可以产生 $s=Router(x)$
其中：$s\in\mathbb{R}^N$ 表示这个 token 对 $N$ 个 experts 的 scores。
然后选出 top-k experts: $\mathrm{TopK}(s)$
>**Note**
>在很多 MoE 里，routing 是 token-level 的。
>也就是每个 token 都会单独决定自己要去哪个 expert，而不是整句话统一送到同一个 expert。

---
## 🌱 Sparsity

MoE 的 sparsity 指：每个 token 只激活全部 experts 中的一小部分。
例如，一个 MoE layer 有 64 个 experts，但每个 token 只选择 2 个 experts：
```math
64 \text{ experts total}
```

```math
2 \text{ experts active per token}
```
这是模型有很多 total parameters，但每个 token 的 active parameters 较少。
>**Note**
>Sparsity 让模型可以扩大参数容量，同时控制每个 token 的计算量。

这也是 MoE 和 dense Transformer 的关键区别：

| **Model Type**   | **每个 token 使用的参数**    |
| ---------------- | --------------------- |
| dense model      | 通常使用整层参数              |
| sparse MoE model | 只使用被 route 到的 experts |

---
## ⚙️ Active Parameters vs Total Parameters

MoE 中需要区分两个参数量：

|**Term**|**Meaning**|
|---|---|
|total parameters|所有 experts 加起来的总参数量|
|active parameters|每个 token 实际激活并参与计算的参数量|
例如，一个 MoE model 可能有：$\text{large total parameters}$，但每个 token 只使用 $\text{small active parameters}$

>**Important**
>MoE 的优势不是“每次使用所有专家”。
>它的优势是：模型整体容量很大，但每个 token 只使用其中一部分。

---
## 🔀 Top-k Routing

Top-k routing 是 MoE 中最常见的 routing 方式之一。
流程可以理解为：
1. 对每个 token 计算 experts scores
2. 选出分数最高的 $k$ 个 experts
3. 只把 token 送到这些 experts
4. 把 experts outputs 按 gate weights 加权组合
形式上:
```math
y=\sum_{i\in\mathrm{TopK}(s)}g_iE_i(x)
```
其中：
- $E_i(x)$ 是第 $i$ 个 expert 的输出
- $g_i$ 是 router / gate 给这个 expert 的权重
- $\mathrm{TopK}(s)$ 是被选中的 experts

>**Note**
>Top-k routing 的直觉：
>每个 token 只找最适合自己的少数几个 experts，而不是让所有 experts 都处理它

---
## 🧠 Why MoE Helps

MoE 的主要动机是：增加模型容量，同时控制 compute cost

Dense model 中，如果增加 FFN 参数，通常每个 token 的计算量也会增加。
MoE 中，可以增加 experts 数量，但每个 token 仍然只激活少数 experts。

因此 MoE 提供了一种 scale model capacity 的方式：
```math
\text{increase total parameters} \quad \text{while keeping active compute relatively small}
```
>**Note**
>这就是为什么 MoE 常被看作一种 "more efficient MLP"
>它不改变 langue modeling objective，而是改变 Transformer block 中 FFN / MLP 的实现方式。

---
## ⚠️ Main Challenges

MoE 的难点不在于概念，而在于 routing 和 systems
#### 1. Load Imbalance
如果 router 把大量 tokens 都送到同一个 expert，而其他 experts 很少被使用，就会出现 load imbalance。例如：10 tokens $\rightarrow$ 5 tokens to expert 1，而其它 experts 只收到很少 tokens。

这会导致：
- 某些 experts 过载
- 某些 experts 学不到东西
- device utilization 不均匀
- batch size 在 expert 层变得不稳定
#### 2. Sparse Training Signal
训练时每个 token 只激活 top-k experts。
没有被选中的 experts 不会看到这个 token，也就不会从这个 token 中获得 gradient signal
>**Note**
>MoE 的 routing 有一点 bandit flavor
>router 只观察被选中的 experts 的结果，却不知道没选中的 experts 如果处理这个 token 会怎样

这也是 MoE 训练比 dense model 更难的原因之一
#### 3. Communication Cost
如果 experts 被放在不同 devices 上，tokens 需要被 route 到对应 device。
这会带来传输 [activations](<../../Neural%20Networks/Activations.md>) 的 communication cost:
```math
\text{route activations} \rightarrow \text{expert computation} \rightarrow \text{return outputs}
```
所以 MoE 不只是 architecture 问题，也和 Systems for Language Models 有关

### 4. MoE 负载均衡 [MoE imbalance mitigation](<./MoE%20imbalance%20mitigation.md>)

#### Per-expert balancing 和 per-device balancing

**Per-expert balancing 关心的是**：每个 expert 收到的 token 数量是否均衡
这个主要是训练角度的问题。如果某些 experts 太少被选中，它们学不到东西；如果某些 experts 太热门，它们过载，而且会加剧 collapse。

**Per-device balancing 关心的是**：不同设备上的负载是否均衡
在真实 MoE 训练里，experts 往往分布在不同 devices 上。即使 per-expert 看起来还行，也可能出现某几台机器上的 experts 特别忙，其他机器比较闲。这样会导致 device utilization 不均匀，大家要等最慢的 device，系统吞吐就差。

所以 DeepSeek 这类系统里，不只是让 experts 均衡，也会让 devices 均衡。这个就是 architecture / training heuristic 和 systems 交叉的地方

---
## 🧪 MoE vs Dense FFN

| **Aspect**  | **Dense FFN**        | **MoE Layer**                      |
| ----------- | -------------------- | ---------------------------------- |
| parameters  | 所有 token 共享同一个 FFN   | 多个 experts                         |
| computation | 每个 token 用同一组 FFN 参数 | 每个 token 只用 top-k experts          |
| capacity    | 参数量和 compute 通常一起增加  | total parameters 可以大幅增加            |
| routing     | no routing           | router / gate 决定 expert assignment |
| challenge   | 相对简单稳定               | routing imbalance、通信、训练复杂度         |

---
## 🚫 Common Confusions

#### MoE 不是多个完整 Transformer

MoE 通常不是复制多个完整 Transformer。
在 LLM 中，MoE 更多是替换 Transformer block 里的 FFN / MLP。

#### MoE 不是每次都用所有 experts

如果每个 token 都用所有 experts，就不是 sparse MoE 的核心思想。
MoE 的关键是每个 token 只激活少数 experts。

#### Router 不等于 expert

Router 决定 token 去哪里；expert 负责真正计算输出。

#### MoE 不等于 ensemble

Ensemble 通常是多个完整模型共同预测。
MoE 是一个模型内部的条件计算结构。

---

>**Summary** — My Understanding
>Mixture of Experts 是一种把 Transformer 中的 FFN / MLP 替换成多个 experts 的 architecture
>
>每个 token 通过 router / gate 选择少数 experts 参与计算，因此 MoE 可以拥有很大的 total parameters，但每个 token 只激活较少 active parameters。
>
>MoE 的核心收益来自 sparsity 和 conditional computation；核心难点来自 routing、load imbalance、sparse training signal 和 expert parallelism。

## **🔗 Connections**

- [Language Model Architecture](<./Language%20Model%20Architecture.md>)
- [Model Architecture](<./Model%20Architecture.md>)
- [Transformer Block](<../../Transformer/Transformer%20Block.md>)
- [Feed-Forward Network](<../../Neural%20Networks/Feed-Forward%20Network.md>)
- [MLP](<../../Transformer/MLP.md>)
