#AI #LanguageModeling

[AdamW](<AdamW.md>) 在实际 implementation 中有个很重要的步骤是 **Bias correction**

⬇️ 的第7行
![adamw](<../Artificial%20Intelligence/attachments/adamw.png>)

这个项实际要分成两个部分考虑：
```math
\frac{1}{1-\beta_1^t}\qquad\frac{1}{1-\beta_2^t}
```
## $m_t$ 展开推导

AdamW 的动量更新公式为：

```math
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
```

假设初始状态 $m_0 = 0$（从第 1 步开始迭代）：
- **当 $t=1$ 时：**$m_1 = \beta_1 \cdot 0 + (1-\beta_1) g_1 = (1-\beta_1) g_1$
- **当 $t=2$ 时：**$m_2 = \beta_1 m_1 + (1-\beta_1) g_2 = \beta_1 (1-\beta_1) g_1 + (1-\beta_1) g_2$
- **当 $t=3$ 时：**$m_3 = \beta_1 m_2 + (1-\beta_1) g_3 = \beta_1^2 (1-\beta_1) g_1 + \beta_1 (1-\beta_1) g_2 + (1-\beta_1) g_3$

提取共同项 $(1-\beta_1)$：$m_3 = (1-\beta_1) \left[ \beta_1^2 g_1 + \beta_1 g_2 + g_3 \right]$

观察规律，对于任意第 $t$ 步，写成累加求和形式就是：
```math
m_t = (1-\beta_1) \sum_{k=1}^{t} \beta_1^{t-k} g_k
```
AdamW 希望 $m_t$ 表示近期 gradients 的加权平均，但是这些 weights 的总和不是 $1$，而是：
```math
1-\beta_1^t
```
早期 $t$ 小，$1-\beta_1^t$ 很小，所以 $m_t$ 会被拉向 $0$，所以 $m_t$ 要除以当前 weight 的总和

```math
\hat m_t = \frac{m_t}{1-\beta_1^t}
```

## $v_t$ 同理
```math
\hat v_t = \frac{v_t}{1-\beta_2^t}
```
