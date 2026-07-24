[Hyperparameters](<../01%20-%20Language%20Modeling%20Basics/Model%20Hyperparameters.md>) choices：

## [Architecture](<../05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>) choice

Architecture 也可以作为 scaling experiment 中被比较的对象。我们可以在小规模比较 LSTM 和 Transformer 的 scaling curves，再决定大规模训练采用哪种 architecture。

用 [Transformer](<../../Transformer/Transformer.md>) 和 LSTMs（一种改进过的RNN） 来举例
>**Question** — 如果有一大笔算力，应该选择 LSTM 还是 Transformer？
>暴力方法：直接训练一个 GPT-3 规模的 LSTM → 贵！！

Scaling-law 方法：
```
训练多个小 LSTM
训练多个小 Transformer
        ↓
在不同 compute budget 下测 loss
        ↓
分别拟合 scaling curves
        ↓
比较 slope 和 intercept
        ↓
预测大规模下谁更好
```

![architecture choice.png](<../../attachments/architecture%20choice.png>)
⬆️ 实验发现 Transformer 的 loss-vs-compute scaling curve 更好，所以我们更有信心把 Transformer 扩大。

## [Optimizer](<../../Transformer/Optimizer.md>) choice

![Adam vs SGD.png](<../../attachments/Adam%20vs%20SGD.png>)
⬆️ 有人做了实验：Adam 并没有改变“增加 compute 后进步多快”，但在每个 compute scale 下都达到更低 loss。因此可以在小规模判断 Adam 的优势可能延续到大规模。

## Depth/Width: Number of layers

#### 模型层数不能太少
![nums of layer.png](<../../attachments/nums%20of%20layer.png>)

- layer 太少会明显限制 scaling；1-layer model 的 loss 更高，下降趋势也更差。
- number of layers 不是 scale-invariant：模型规模增大时，合适的 depth 通常也要增加。

#### 多深、多宽才合适？

在参数总量差不多的情况下，可以有不同设计：
- 少量很宽的层
- 很多较窄的层
- 中等层数 + 中等宽度
![aspect ratio.png](<../../attachments/aspect%20ratio.png>)

- 更适合跨规模迁移的是 **aspect ratio**（$d_{model}/n_{layer}$），而不是固定的 layer 数量。
- 不同 model size 的最佳 aspect ratio 落在相近区域，而且最优点附近通常比较平坦：一段范围内的 depth/width 配置表现接近。
- 不需要精确找到唯一的“完美组合”。在一个比较大的合理范围内，不同 depth/width 组合的表现都差不多。
#### parameter count 的计算方法很重要

Transformer 里的 parameters 不只有 Transformer blocks，还包括 embedding table。
![not all parameters are made equal.png](<../../attachments/not%20all%20parameters%20are%20made%20equal.png>)

>**Important**
> Not all parameters are created equal. Embedding parameters 与 Transformer block 中参与主要计算的 parameters 作用不同；是否把 embedding 算进 parameter count，会明显改变 scaling curve。因此比较 depth/width 时必须先统一 $N$ 的定义。

## [Batch Size](<../01%20-%20Language%20Modeling%20Basics/Batch%20Size.md>): Critical batch size

critical batch size 本身也呈现可预测的 scaling behavior：
![critical batch size.png](<../../attachments/critical%20batch%20size.png>)

- 横轴：Train Loss --- 从左到右数值越来越小
- 纵轴：Critical Batch Size (log-scale) --- 越往上，表示在出现明显递减收益之前，可以使用越大的 batch。
⬆️ 蓝线和橙线总体都从左下走向右上： **Target loss 越低，critical batch size 越大。** critical batch size 会随着目标 loss 降低而有规律地增大。

## [Learning rates](<../../Transformer/Learning%20Rate%20Schedule.md>): μP and scale-aware LR choices
![learning rate scaling.png](<../../attachments/learning%20rate%20scaling.png>)
⬆️ 代表了两种不同的 learning rate 选择策略
⬅️ ****scale-aware****: Intuitively, the bigger my model the smaller learning rate should be. Because the bigger more parameters, more things are changing at once. **Move less**

➡️ ****μP**** 的目标是通过调整：parameter initialization；不同 parameter tensor 的更新缩放；optimizer step 的 scale；让不同 width 的模型具有相近的训练动态。
因此右图中，不同 width 的 U 形曲线基本重合，最低点也在相近位置：
