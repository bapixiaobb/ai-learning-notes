#LanguageModeling #Transformer #PositionEncoding #Implementation

这张卡记录 [[Rotary Position Embedding]] 的 implementation 思路。概念直觉见 [[Rotary Position Embedding]]。

>[!note]
> RoPE implementation 的核心是: 沿最后一维拆 even / odd columns, 对每一对 hidden dimensions 套二维旋转公式, 再拼回原 shape。

## Input Shape

作业里的 RoPE 接口大致是:

```python
forward(x, token_positions)
```

其中:

```text
x.shape = [..., seq_len, d_k]
token_positions.shape = [..., seq_len]
```

可以先把它想成:

```text
x.shape = [batch, seq_len, d_k]
```

含义是:

```text
batch   = 一批 sequence
seq_len = token rows
d_k     = 每个 query/key vector 的 hidden dimension columns
```

>[!important]
> RoPE 拆的是最后一维 $d_k$, 不是拆 sequence/token 这一维。

## Even / Odd 拆法

如果某个 token 的 query/key vector 是:

```text
[a, b, c, d]
```

RoPE 把它分成:

```text
(a, b), (c, d)
```

Python slicing:

```python
x_even = x[..., 0::2]
x_odd = x[..., 1::2]
```

对应:

```text
x_even = [a, c]
x_odd  = [b, d]
```

这里的 `...` 表示前面的 batch / sequence 等维度都保留, 只在最后一维上切。

## Frequency

RoPE 的 frequency 是:

```math
\omega_i = \theta^{-2i/d_k}
```

其中:

```text
i = 0, 1, ..., d_k/2 - 1
```

代码:

```python
pair_idx = torch.arange(0, d_k // 2, device=x.device)
freqs = theta ** (-2 * pair_idx / d_k)
```

这和公式里的

```math
\frac{1}{\Theta^{(2k-2)/d}}
```

是同一个东西。作业里 $k$ 从 1 开始, Python 里的 `pair_idx` 从 0 开始。

## Angle Table

RoPE 角度是:

```math
\text{angle} = \text{position} \times \text{frequency}
```

代码:

```python
angles = token_positions[..., None].to(x.device) * freqs
```

如果:

```text
token_positions.shape = [seq_len]
freqs.shape = [d_k/2]
```

那么:

```text
token_positions[..., None].shape = [seq_len, 1]
angles.shape = [seq_len, d_k/2]
```

这得到的是一张表:

```text
每个 token position × 每个 dimension pair 的旋转角度
```

然后:

```python
cos = torch.cos(angles)
sin = torch.sin(angles)
```

## 写法 1: 显式拆 Even / Odd

>[!note]
> 这个写法最接近二维旋转公式, 适合理解。

```python
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_k
        assert self.d_k % 2 == 0

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        pair_idx = torch.arange(0, self.d_k // 2, device=x.device)
        freqs = self.theta ** (-2 * pair_idx / self.d_k)

        angles = token_positions[..., None].to(x.device) * freqs
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out
```

对应公式:

```math
a' = a\cos\alpha - b\sin\alpha
```

```math
b' = a\sin\alpha + b\cos\alpha
```

## 写法 2: rotate_half

>[!note]
> 这个写法更短, 本质上是把 even / odd 的正负号打包进 `rotate_half`。

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out = torch.empty_like(x)
    out[..., 0::2] = -x_odd
    out[..., 1::2] = x_even
    return out
```

`rotate_half` 做的是:

```text
[a, b, c, d] -> [-b, a, -d, c]
```

如果把 cos/sin 扩展回完整 hidden dimension:

```python
out = x * cos + rotate_half(x) * sin
```

这等价于:

```text
[a*cos0 - b*sin0,
 b*cos0 + a*sin0,
 c*cos1 - d*sin1,
 d*cos1 + c*sin1]
```

完整写法:

```python
class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.d_k
        assert self.d_k % 2 == 0

        pair_idx = torch.arange(0, self.d_k // 2, device=x.device)
        freqs = self.theta ** (-2 * pair_idx / self.d_k)

        angles = token_positions[..., None].to(x.device) * freqs
        cos_half = torch.cos(angles)
        sin_half = torch.sin(angles)

        cos = torch.empty_like(x)
        sin = torch.empty_like(x)
        cos[..., 0::2] = cos_half
        cos[..., 1::2] = cos_half
        sin[..., 0::2] = sin_half
        sin[..., 1::2] = sin_half

        return x * cos + rotate_half(x) * sin
```

## max_seq_len 和 Buffer 优化

简单版本每次 forward 都现算:

```text
pair_idx -> freqs -> angles -> cos/sin
```

所以 `max_seq_len` 没有真正用到。

作业提示的优化是: 在 `__init__` 里提前把所有 position 的 cos/sin 算好。

```text
positions = 0, 1, ..., max_seq_len - 1
cos_cached.shape = [max_seq_len, d_k/2]
sin_cached.shape = [max_seq_len, d_k/2]
```

然后 forward 时根据 `token_positions` 查表:

```python
cos = self.cos_cached[token_positions]
sin = self.sin_cached[token_positions]
```

这些 cos/sin 不是 learnable parameters, 所以不应该用 `nn.Parameter`。

应该用 buffer:

```python
self.register_buffer("cos_cached", cos, persistent=False)
self.register_buffer("sin_cached", sin, persistent=False)
```

>[!note]
> Buffer 会跟着 module `.to(device)` 移动, 但不会被 [[Optimizer]] 更新。
>
> `persistent=False` 表示它不存进 `state_dict`, 因为 cos/sin 可以重新计算出来。

预计算版本的 init 大致是:

```python
pair_idx = torch.arange(0, d_k // 2, device=device)
freqs = theta ** (-2 * pair_idx / d_k)

positions = torch.arange(0, max_seq_len, device=device)
angles = positions[:, None] * freqs[None, :]

self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
```

## Self Check

>[!tip]
> 1. Position 0 应该是恒等变换: angle = 0, cos = 1, sin = 0, 输出等于输入。
> 2. 输出 shape 应该和输入 shape 完全一样。
> 3. 拆的是最后一维 columns, 不是 token rows。
> 4. RoPE 没有 learnable parameters; cos/sin 是固定公式生成的。

## Related

- [[Rotary Position Embedding]]
- [[Query Key Value]]
- [[Self-Attention]]
- [[Positional Encoding]]
