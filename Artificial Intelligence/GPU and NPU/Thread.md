#GPU

## What is thread
>**Example** — 假设要计算 $y=x^2$，长度：100万个元素。
>CPU 需要一个元素一个元素的算

[GPU](<GPU.md>) 做法是 100万个工人，同时干。一个 Thread 可以理解成 一个工人。
（GPU 会 launch 很多 threads，但不是所有 threads 真正同一时刻都在运行；它们会以 thread blocks / warps 的形式分批调度到 [SM](<Streaming%20Multiprocessor.md>) 上。）

>**Note** — 每个 thread 都有：
>- 自己的 `threadIdx`
>- 自己的寄存器状态
>- 自己计算出的地址
>- 自己的逻辑控制流
>- 自己应该处理的数据
#### Why need thread block?

说是能让 threads 协作，其实就是为了让一组 threads 能在同一个 SM 上共享 shared memory

>**Example**
> 对 elementwise GeLU，一个 thread 算一个元素就够了。但 softmax 需要整行的 max/sum，matmul 需要复用 A/B tile，所以 threads 需要协作；thread block 就是这组协作 threads 的边界。
