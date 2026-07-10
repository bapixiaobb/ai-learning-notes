#AI #LanguageModeling #GPU 

一种 [[GPU Memory Bound]] 的优化方法

举个例子
普通 [[Backpropagation]] 是 forward: x → a1 → a2 → a3 → out，把 a1, a2, a3 都存下来，然后在 backward 的时候，用 a1, a2, a3 算梯度

但是 [[GPU]] 上存 activation 很贵，尤其是大模型训练里，activation memory 非常大。

recomputation 的做法就是不存 a1, a2, a3，只存必要的东西，然后在 backward 的时候，需要 a1 的时候，从 x 重新算 a1；需要 a2 的时候，再重新算 a2

这样结果就是：多做 compute，少存 activation，少读写 memory

相当于用计算换内存