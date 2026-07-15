#AI #LanguageModeling #GPU

为了解决 [Pipeline Parallelism](<./Pipeline%20Parallelism.md>) 中 bubble 的问题，这是一种特殊的调度技巧

就是把 [backwards](<../../Neural%20Networks/Backpropagation.md>) 拆成 B 和 W
1. Backpropagating activation (z, x)
2. Computing weight gradients (W) --- for [optimizer state](<../../Transformer/Optimizer.md>)

## Example

对于一层简单的计算：
```math
z=xW
```
backward 要计算两种 gradient:
```math
\frac{\partial L}{\partial x}\,\,\,\,\,\,\,\,\, \,\,\,\,\,\,\,\,\,\text{和}\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\, \frac{\partial L}{\partial W}
```
它们用途不同：
```
B：计算 ∂L/∂x  → 必须立刻传给前一个 pipeline stage
W：计算 ∂L/∂W  → 只要 optimizer step 之前算完就行
```

## How it works?

其中 `B` 在关键依赖链上：
```
Stage 3 算出 ∂L/∂x
        ↓
Stage 2 才能开始 backward
        ↓
Stage 1 才能开始 backward
```

但是 `W` 不需要传给前一个 stage。前一个 stage 的 backward 并不关心这一层的 parameter gradient，所以它可以晚一点算。

Zero Bubble 把它拆开：
```
[B]     [W]
 紧急     可以推迟
```

![Computation Graph for MLP.png](<../attachments/Computation%20Graph%20for%20MLP.png>)

调度器先做 `B`，赶紧把 activation gradient 传给前一张卡；暂时不做的 `W` 放进后面原本空闲的白色 bubble：

```
原来： F  B  ·  ·  F  B  ·
现在： F  B  W  W  F  B  W
```

## Cost

代价是实现非常复杂，而且推迟 `W` 往往需要更久地保留相关 intermediates，可能增加显存压力；还需要改动 auto diff 的执行顺序和 optimizer synchronization。