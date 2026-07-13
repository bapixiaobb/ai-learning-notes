#AI #LanguageModeling #GPU

一种 [GPU Memory Bound](<./GPU%20Memory%20Bound.md>) 的优化方法
# Intuition

Think of a [GPU](<./GPU.md>) like a factory – inputs come from a warehouse (memory) and is processed at a factory
![GPU factory.png](<../attachments/GPU%20factory.png>)
如果我们需要做很多 operations 的话，一直把数据从 memory 搬到 compute 再搬回 memory 就显得笨笨的。所以我们可以把多个 operations 写成一个 kernel，这样就只用搬两次![Fused Kernel.png](<../attachments/Fused%20Kernel.png>)