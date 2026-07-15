#DeepLearning #NeuralNetwork #ComputationalGraph

Computational Graph（也常写作 computation graph）描述一次计算中：**哪些 tensors 由哪些 operations 得到，以及它们彼此怎样依赖。**

例如：

```math
x
\rightarrow
z=Wx+b
\rightarrow
a=\sigma(z)
\rightarrow
\hat y
\rightarrow
\mathcal L
```

它是一张“计算依赖地图”，不是 forward、backward 或 parameter update 本身。

## 一次 Training Step 怎样经过这张图

1. [Forward pass](<./Forward%20Propagation.md>) 沿依赖方向计算各个 tensor 的值，得到 model output 和 intermediate [Activations](<./Activations.md>)。
2. Training 时，loss function 再根据 model output 和 targets 计算 loss。
3. [Backward pass](<./Backpropagation.md>) 从 loss 出发，沿依赖关系反向应用 chain rule，计算 gradients。
4. [Optimizer step](<../Transformer/Optimizer.md>) 读取 parameter gradients，更新 parameters。

>**Important**
> Forward 负责算 values，backward 负责算 gradients，optimizer 才负责更新 parameters。

## Architecture 和 Runtime Graph

[Model Architecture](<../Language%20modeling/05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>) 决定 model forward 的基本计算结构，例如 attention、MLP、normalization 和 residual connection 怎样连接。

实际某一次运行的 computation graph 由这次真正执行的 operations 构成。Training 会把 loss 接到 model output 后面；dropout、train / eval mode 或条件分支也可能让实际执行路径有所不同。

因此，architecture 和 computation graph 联系紧密，但不是完全相同的词。

## PyTorch

当 gradient tracking 开启，并且相关 tensors 需要 gradients 时，PyTorch 会在 forward 中动态记录 tensor operations 及其依赖关系。

    loss.backward()

会沿这张 graph 反向计算 gradients；之后：

    optimizer.step()

才使用 parameter gradients 更新 parameters。

>**Summary**
> Computation graph 是依赖关系；forward 为它计算 values，backward 为它计算 gradients，optimizer 使用 gradients 改变 parameters。

## Related

- [Forward Propagation](<./Forward%20Propagation.md>)
- [Backpropagation](<./Backpropagation.md>)
- [Activations](<./Activations.md>)
- [Optimizer](<../Transformer/Optimizer.md>)
- [Model Architecture](<../Language%20modeling/05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>)
