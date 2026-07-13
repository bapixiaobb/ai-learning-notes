>**Note** — Triton 是一种用于编写 [GPU](<./GPU.md>) kernel 的语言和编译器
>它允许程序员从 block / tile 的角度描述计算，再由 compiler 将其映射到具体的 threads、warps、registers 和 GPU instructions


## Cuda 和 Triton 有什么不同

>**Important**
>CUDA 和 Triton 最终都要生成 GPU 可以执行的底层指令。区别主要不在“能算什么”，而在程序员描述计算时所处的抽象层级。

### Cuda: Thread-level Programming

>**Note**
> CUDA kernel 通常描述：**每个 thread 做什么**。
>
> 程序员通过 thread ID 找到这个 thread 应该处理的数据，并更显式地管理 shared memory、线程同步和数据协作。

```
找到 thread ID
→ 计算这个 thread 的数据地址
→ load
→ compute
→ store
```

### Triton: Block-level Programming

>**Note**
>Triton kernel 通常描述：**一个 program instance 处理哪一个 data block / tile，以及如何对整个 block 进行计算**。

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
x = tl.load(x_ptr + offsets)
y = gelu(x)
tl.store(y_ptr + offsets, y)
```
这里的 `x` 可以看到，是一组 values，而不是单个 scalar