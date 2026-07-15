#AI #LanguageModeling

[Flash Attention](<./Flash%20Attention.md>) 中的 [Softmax](<./Softmax.md>)

>**Note** — Online softmax
>To keep track of the max, incrementally update the max, and set up a telescoping sum
>This lets you compute the softmax tile-by-tile

![online softmax.png](<./online%20softmax.png>)

一边扫 tile，一边更新：当前见过的最大值 $\max$，当前积累的 $\sum e$，当前累积的输出 $O$。
当新的 tile 里出现更大的 $\max$ 时，就把之前的累积量按比例 rescale

# Implementation

在 [Query Key Value](<./Query%20Key%20Value.md>) 中
```math
\mathrm{softmax}(\frac{QK^\top}{\sqrt{d}})V
```

在 [Flash Attention](<./Flash%20Attention.md>) 中 online softmax 输入一般接收的是 `qkVecTile`，我们设 `scoreTile = qkVecTile` `scale = 1 / sqrt(d)`

```math
\rightarrow\frac{QK^\top}{\sqrt{d}}=\text{scoreTile}\times scale=(s_i \times scale)
```

```math
\rightarrow\mathrm{softmax}(s_i \times scale)\times V_i=\frac{e^{(s_i \times scale)-\max(s_i \times scale)}\times V_i}{\sum_{j=1}^{K} e^{(s_i \times scale)-\max(s_i \times scale)}}
```
因为 $scale = 1 / \sqrt{d}$ 是正数，所以：
```math
\max(s \times scale) = \max(s) \times scale
```
所以：
```math
e^{(s_i \times scale)-\max(s_i \times scale)}=e^{(s_i - M)\times scale}
```

```math
\rightarrow\mathrm{softmax}(s_i \times scale)\times V_i
=
\frac{e^{(s_i - M)\times scale}\times V_i}
{\sum_j e^{(s_j - M)\times scale}}
```
其中 $M = \max_j(s_j)$

```python
def online_softmax_attention(score_tiles, value_tiles, scale):
    """s
    score_tiles: QK^T 沿 key/sequence 维度切成的多个 tile
                 每个 score_tile 可以理解成 [q_rows, kv_tile]
    value_tiles: 和 score_tiles 一一对应的 V tile
                 每个 value_tile 可以理解成 [kv_tile, head_dim]
    scale:       1 / sqrt(d)

    return:
        out: softmax(QK^T * scale) @ V 的当前 q_rows 输出
    """
    m = -inf      # running max, shape: [q_rows, 1]
    l = 0.0       # running sum(exp), shape: [q_rows, 1]
    out = 0.0     # running output, shape: [q_rows, head_dim]

    for score_tile, value_tile in zip(score_tiles, value_tiles):
        # 1. 当前 tile 每一行的最大值
        tile_m = max(score_tile, axis=-1, keepdim=True)

        # 2. 更新到目前为止见过的最大值
        new_m = maximum(m, tile_m)

        # 3. 旧累积量的 rescale 系数
        #    把基于 old m 的累积量，改写到基于 new_m 的尺度下，rescale 一般 <= 1
        rescale = exp((m - new_m) * scale)

        # 4. 当前 tile 的未归一化 softmax 分子
        p = exp((score_tile - new_m) * scale)

        # 5. 更新 softmax 分母
        tile_l = sum(p, axis=-1, keepdim=True)
        new_l = l * rescale + tile_l

        # 6. 当前 tile 对 attention 输出的贡献
        tile_out = p @ value_tile

        # 7. 更新累计输出；最后再统一除以分母
        out = out * rescale + tile_out

        # 8. 写回 running state
        m = new_m
        l = new_l

    # 9. 最终归一化
    return out / l
```

# Operation Implementation

一般来说算子实现中 cube 和 vector 计算的流水是分开的，所以 `p @ value_tile` 这种矩阵乘前后的 vector 操作需要分隔开
## Online Softmax 实现拆分

这版实现把 FlashAttention 里的 online softmax 拆成三步：

1. online_softmax_tile：只处理当前 score tile，得到局部 `p` / `tile_m` / `tile_l`
2. cube：用当前 `p` 去乘当前 `value_tile`，得到当前 tile 对 O 的未归一化贡献 `tile_out`
3. update_state：把历史累计状态和当前 tile 状态合并

注意：
- online_softmax_tile 里面的 pij 是相对 `tile_m` 算的，不是相对全局 `new_m` 算的。
- 跨 tile 的 `new_m` 对齐发生在 update_state 里面。

输入约定：
`score_tile` : 当前 `QK^T` block，shape `[K, Q]`
`value_tile` : 当前 `V` block，shape `[K, D]`

历史状态：
`state_m` : 已处理 block 的 running max，shape `[1, Q]`
`state_l` : 已处理 block 的 running sum，shape `[1, Q]`
`state_o` : 已处理 block 的未归一化输出累计，shape `[D, Q]`


```python
def online_softmax_tile(score_tile, scale):
    """
    score_tile: 当前 QK^T tile，可以理解成 [q_rows, kv_tile]
    scale:      1 / sqrt(d)

    return:
        p:      当前 tile 的未归一化 softmax 分子
        tile_m: 当前 tile 的 row max
        tile_l: 当前 tile 的 row sum(exp)
    """

    # 1. 先乘 scale
    scaled_score = score_tile * scale

    # 2. 当前 tile 每一行自己的最大值
    tile_m = max(scaled_score, axis=-1, keepdim=True)

    # 3. 当前 tile 的未归一化 softmax 分子
    p = exp(scaled_score - tile_m)

    # 4. 当前 tile 自己的局部分母
    tile_l = sum(p, axis=-1, keepdim=True)

    return p, tile_m, tile_l


def online_softmax_update(m, l, out, tile_m, tile_l, tile_out):
    """
    m:        running max, shape: [q_rows, 1]
    l:        running sum(exp), shape: [q_rows, 1]
    out:      running output, shape: [q_rows, head_dim]

    tile_m:   当前 tile 的 max
    tile_l:   当前 tile 的 sum(exp)
    tile_out: 当前 tile 的 p @ value_tile
    """

    # 1. 更新到目前为止见过的最大值
    new_m = maximum(m, tile_m)

    # 2. 旧累积量的 rescale
    #    把基于 old m 的 l/out，换到 new_m 的尺度下
    old_rescale = exp(m - new_m)

    # 3. 当前 tile 累积量的 rescale
    #    把基于 tile_m 的 tile_l/tile_out，换到 new_m 的尺度下
    tile_rescale = exp(tile_m - new_m)

    # 4. 更新 softmax 分母
    new_l = l * old_rescale + tile_l * tile_rescale

    # 5. 更新累计输出
    new_out = out * old_rescale + tile_out * tile_rescale

    return new_m, new_l, new_out


def online_softmax_attention(score_tiles, value_tiles, scale):
    """
    score_tiles: QK^T 沿 key/sequence 维度切成的多个 tile
                 每个 score_tile 可以理解成 [q_rows, kv_tile]

    value_tiles: 和 score_tiles 一一对应的 V tile
                 每个 value_tile 可以理解成 [kv_tile, head_dim]

    scale:       1 / sqrt(d)

    return:
        out: softmax(QK^T * scale) @ V 的当前 q_rows 输出
    """

    m = -inf      # running max, shape: [q_rows, 1]
    l = 0.0       # running sum(exp), shape: [q_rows, 1]
    out = 0.0     # running output, shape: [q_rows, head_dim]

    for score_tile, value_tile in zip(score_tiles, value_tiles):
        # 1. 当前 tile 的 online softmax 局部统计
        p, tile_m, tile_l = online_softmax_tile(score_tile, scale)

        # 2. 当前 tile 对 attention 输出的贡献
        tile_out = p @ value_tile

        # 3. 合并历史 state 和当前 tile state
        m, l, out = online_softmax_update(
            m, l, out,
            tile_m, tile_l, tile_out,
        )

    # 4. 最终归一化
    return out / l
```

# Intuition

可以把完整 attention 写成：
```math
O =\frac{\sum (e^{(score_i * scale - M)} * V_i )}{\sum e^{(score_i * scale - M)} }
```

[Flash Attention](<./Flash%20Attention.md>) / online softmax 就是把它拆成两条累计线：

```
分子累计:
N = Σ [ exp(score_i * scale - M) * V_i ]

分母累计:
L = Σ [ exp(score_i * scale - M) ]
```
因为最终的 M 一开始不知道，所以每来一个 tile 都会 rescale `N` 和 `L`，最后才做：`O = N / L
`
❌ 所以不是每个 tile 都算一个局部 softmax
✅ 而是把 softmax 的计算和，外面的 `* V_i` 揉在一起，最后再归一化