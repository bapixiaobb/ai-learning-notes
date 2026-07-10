#AI #LanguageModeling 

## What is Decoding

在 [[Training vs Inference]] 中有介绍 [[Language Modeling]] 里的两个步骤，先是训练模型。模型训练完成之后，我们可以从我们模型中生成 text

从 [[Transformer]] 最后一张图可以看出来，最后模型生成的是一个 `[batch, seq, vocab_size]` 这样的 matrix，每个 sequence position 都对应一个长度为 `vocab_size` 的 logits vector。它是未归一化分数，经过 Softmax 后才成为 [[Next-token prediction]] 的 distribution。
![[TransformerLM.jpeg]]
它完整的步骤是，输入 prompt 之后，根据这个 prompt 来预计 ▶️ token，也就是：

```math
x_1, \dots, x_T
\rightarrow
x_{T+1}
```

然后把新 token 接回输入：

```math
x_1, \dots, x_T, x_{T+1}
\rightarrow
x_{T+2}
```

继续重复：

```math
x_1, \dots, x_T, x_{T+1}, x_{T+2}
\rightarrow
x_{T+3}
```
从上面的步骤就可以看出，我们每次其实只需要最后一个 position 的 [[Next-token prediction]]. 

## Single Decoding Step

所以我们把这个 prompt 作为输入放进我们训练好的模型跑一圈之后，得到
```
logits = model(input_tensor)
```
只需要取最后一个 `seq` 的得分就好
```
next_token_logits = logits[0, -1, :]
```

得到最后一个 position 对下一个 token 的 logit 之后，我们需要把这个 logit 通过 [[Softmax]] 转换成概率分布

假设 sequence $x_{1...t}$ 是我们的输入，输出是 $x_{t+1}$，那一个简易的 decoder 模型是 
```math
P(x_{t+1}=i\mid x_{1:t})
=
\frac{\exp(v_i)}
{\sum_j\exp(v_j)}
```
```math
v=\operatorname{TransformerLM}(x_{1:t})[-1]
\in\mathbb{R}^{\text{vocab\_size}}
```
## Temperature scaling

这是针对 [[Softmax]] 的一个简单的 scaling 小技巧
```math
\mathrm{softmax}
=
\frac{\exp(v_i/\ \tau)}{\sum_j \exp(v_j/ \tau)}
```
看两个 token 的概率比：
```math
\frac{p_i}{p_j}
=
\exp\left(\frac{v_i-v_j}{\tau}\right)
```
```
τ < 1：放大 logits 之间的差距，分布更尖锐
τ = 1：保持原分布
τ > 1：缩小相对差距，分布更平坦
τ → 0：最大 logit 对应的 token 占据几乎全部概率
```

## Top-p Sampling

Modify the sampling distribution by truncating low-probability tokens.

假设上一步 softmax 得到的结果是一个叫 $q$ 的 probability distribution，$p$ 是一个 hyperparameter

```math
P(x_{t+1}=i|q)=\left\{\begin{matrix}
\frac{q_i}{\sum_{j\in V(p)}q_j}   & \text{if }i\in V(p)  \\
 0 & \text{otherwise} 
\end{matrix}\right.
```

这个 $V(p)$ 是 smallest set of indices such that $\sum_{j\in V(p)}q_j\geq p$

怎么解释呢，举一个例子
>[!example] 首先要排序，给定排序好的概率分布：`q = [0.50, 0.35, 0.10, 0.05]`
> 累计概率：
>```
> C = cumsum(q) = [0.50, 0.85, 0.95, 1.00]
>```
> 设 `top_p = 0.8`
> 最小的集合应该保留：`[0.50, 0.35]`
> 因为
>```
> 0.50 < 0.8
> 0.50+0.35=0.85 ≥ 0.8
>```
>关键是：第二个 token 虽然让累计概率超过了 `0.8`，但它必须保留，因为它是“达到 threshold 的那个 token”。
>
>**具体实现**
>```
>C:       [0.50, 0.85, 0.95, 1.00]
>q:       [0.50, 0.35, 0.10, 0.05]
>C-q:     [0.00, 0.50, 0.85, 0.95]
>```
>比较：`remove_mask = (C - q) >= 0.8`
>返回：`[False, False, True, True]`
>含义：
>```
>第 0 个：加入它之前是 0，还不能删除
>第 1 个：加入它之前是 0.5，还不能删除
>第 2 个：加入它之前已经是 0.85，可以删除
>第 3 个：加入它之前已经是 0.95，可以删除
>```

## Autoregressive Loop

```
prompt
→ tokenizer.encode
→ token IDs
→ 添加 batch dimension
→ model.eval() + no_grad()
→ model forward
→ logits[0,-1,:]
→ temperature
→ softmax
→ top-p
→ multinomial sample
→ append next token ID
→ EOS 或 max_new_tokens 时停止
→ tokenizer.decode
```

## 和 [[Tokenization]] 的 decode 有什么区别

- tokenization 中的 decode 就是 `token IDs → text`
- autoregressive decoding：反复预测、采样并追加 next token 的完整 inference process

## Naive Decoding and KV Cache

TODO