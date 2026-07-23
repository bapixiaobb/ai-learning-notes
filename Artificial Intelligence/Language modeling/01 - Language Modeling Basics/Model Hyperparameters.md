#AI #LanguageModeling

[LMs](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20%28LLM%29.md>) 训练前或训练过程中设定的配置，不是由训练学习的。**是决定“模型如何搭建、如何训练”的外部配置。**

例如一个 Linear 层：
```math
y=xW+b
```
这些是 hyperparameters：

- step size：`3e-4`
- batch size：`256`
- number of layers：`24`
- hidden dimension：`2048`
- number of attention heads：`16`
- optimizer：`AdamW`
- activation function：`SwiGLU`
- weight decay：`0.1`
这些都是训练前指定的，不会随着训练更新

所以 `hyper` 的意思不是“高维”，而是**位于普通模型参数之上的配置层级**。

> **Note** — Hyperparameter space is high-dimensional 表示：
>
>```
>(step size, batch size, depth, width, heads, optimizer, ...)
>```
>有很多个可以同时调整的坐标，因此搜索空间是多维的。这里的“多维”描述的是**配置空间**，不是说所有 hyperparameters 被储存在一个大 tensor 中。


```
hyperparameters
    ↓ 决定
模型结构和训练方式
    ↓ 产生并更新
parameter tensors
```

例如：

```
model = TransformerLM(
    d_model=2048,   # hyperparameter
    num_layers=24,  # hyperparameter
)
```

创建模型以后，内部才会产生大量 parameter tensors：

```
embedding.weight
layer[0].q_proj.weight
layer[0].k_proj.weight
...
```
