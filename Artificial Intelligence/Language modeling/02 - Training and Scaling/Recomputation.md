#AI #LanguageModeling #GPU

一种 [GPU Memory Bound](<../03%20-%20GPU%20and%20Systems/GPU%20Memory%20Bound.md>) 的优化方法

一次普通 training step 会：

1. 通过 [forward pass](<../../Neural%20Networks/Forward%20Propagation.md>) 计算 $x \rightarrow a_1 \rightarrow a_2 \rightarrow a_3 \rightarrow out$；
2. 暂时保存 [backward pass](<../../Neural%20Networks/Backpropagation.md>) 需要的一部分 [activations](<../../Neural%20Networks/Activations.md>)；
3. backward 使用这些中间结果计算 gradients。

不是每个 intermediate tensor 都必须保存，但保存 backward 所需的数据仍然可能占用大量 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) memory。

Recomputation 的做法是只保存少量 checkpoints。Backward 需要某一段的 intermediate activations 时，就从最近的 checkpoint 重新执行那一段 forward。

这样结果就是：多做 compute，少存 activation，少读写 memory

相当于用计算换内存
