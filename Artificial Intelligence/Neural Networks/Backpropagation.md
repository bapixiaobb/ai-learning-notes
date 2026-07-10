#DeepLearning #NeuralNetwork #Optimization #ComputationalGraph

Backpropagation 是从 loss 出发，沿 [[Computational Graph]] 反向应用 chain rule，计算 loss 对模型参数 gradients 的过程。

它负责**计算梯度**，不负责更新参数。

对应 PyTorch：

```python
optimizer.zero_grad()

logits = model(inputs)
loss = loss_function(logits, targets)

loss.backward()
optimizer.step()
```

- `loss.backward()`：计算 gradients；
- `optimizer.step()`：更新 parameters；
- `optimizer.zero_grad()`：清除上一次留下的 gradients。

## Chain Rule

假设 computation graph 中有：

$$ x \rightarrow y \rightarrow \mathcal{L} $$

其中：

$$ y=f(x) $$

backward 时，我们已经从后面的计算得到：

$$ \frac{\partial \mathcal{L}}{\partial y} $$

再结合当前操作的局部导数：

$$ \frac{\partial y}{\partial x} $$

便可以计算：

$$ \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial y}{\partial x} $$

所以 backward 中传递的不是笼统的“误差”，而是 upstream gradient。每个操作把 upstream gradient 与自己的局部导数结合，再传给前面的变量。

## Scalar Example

假设：

$$ z=wx $$$$ \mathcal{L}=z^2 $$

先计算：

$$ \frac{\partial \mathcal{L}}{\partial z}=2z $$

以及：

$$ \frac{\partial z}{\partial w}=x $$

根据 chain rule：

$$ \frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial z} \frac{\partial z}{\partial w} = 2zx $$

这里：

- $2z$ 是从 loss 传来的 upstream gradient；
- $x$ 是乘法操作对 $w$ 的 local derivative；
- $2zx$ 是最终得到的 parameter gradient。

## PyTorch Autograd

在 forward 中，PyTorch 会记录 tensor operations 及其关系，形成 computation graph。

调用：

```
loss.backward()
```

之后，模型参数的梯度会存入：

```
parameter.grad
```

例如：

```
print(model.lm_head.weight.grad)
```

PyTorch 默认会累积 gradients，因此每轮训练通常需要调用：

```
optimizer.zero_grad()
```

## Backpropagation vs Optimizer

> [!important]  
> Backpropagation 和 [[Optimizer]] 不是同一件事。

Backpropagation 计算：

$$ \nabla_\theta \mathcal{L} $$

Optimizer 使用这个 gradient 更新参数，例如 [[Gradient Descent]]：

$$ \theta \leftarrow \theta-\eta\nabla_\theta\mathcal{L} $$

其中 $\eta$ 是 learning rate。

## In Transformer

Transformer 的 forward 包含：

```
Embedding
→ Attention
→ FFN
→ LM Head
→ Loss
```

只要这些计算由 PyTorch tensor operations 构成，autograd 就能从 loss 一直反向计算到所有 `nn.Parameter`。

因此我们不需要为 Attention、RMSNorm 或 SwiGLU 手写 backward。

为了进行 backward，PyTorch 通常需要保存 forward 中的中间 activations。[[Recomputation]] 或 activation checkpointing 会选择少保存一部分 activations，并在 backward 时重新计算，以交换 memory 和 compute。

> [!summary]  
> Backpropagation 是在 computation graph 上反向应用 chain rule，计算 loss 对参数的 gradients。
> 
> `loss.backward()` 计算并保存 gradients，`optimizer.step()` 才真正更新 parameters。

## Connections

- [[Forward Propagation]]
- [[Computational Graph]]
- [[Automatic Differentiation]]
- [[Optimizer]]
- [[Gradient Descent]]
- [[Recomputation]]