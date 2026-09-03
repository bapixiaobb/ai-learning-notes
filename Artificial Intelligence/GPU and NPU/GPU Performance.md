#GPU

**[GPU](<GPU.md>) 优化有两条主线：怎么组织计算，以及怎么组织数据。**
## Execution Model?

假设：Warp A 需要读 HBM，HBM 很慢，可能需要 500 cycles 以后才回来
GPU 不会像 CPU 一样等，而是准备了很多 Warp，在等待内存时不停切换工作。

GPU 不像 CPU 那样优化单线程 latency，而是用很多 [Thread](<Thread.md>) / warp 提高 throughput: [GPU 核心思想](<GPU%20%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3.md>)：[SIMT](<SIMT.md>) / occupancy / warp scheduling

## Reuse

假设计算 $C=AB$ 如果每次计算 $A_{ik}$ 都要去 HBM 重新读，就会很慢。

正确做法应该是 HBM -> Shared Memory 搬运一次，然后重复使用很多次，这样搬运成本被摊薄了。这就引出了 GPU 的 [GPU Memory Bound](<GPU%20Memory%20Bound.md>)

所以 thread block 的意义就是：
> 把一组需要合作的 threads 绑在一起，并保证它们能在同一个 SM 上共享 shared memory

>**Note** — 所以 GPU 优化核心变成：
>减少搬运
 如何提高 Data Reuse。
 >目的就是：**让昂贵的 compute unit 不要等数据。**

## [SM](<Streaming%20Multiprocessor.md>) 上的 warp 够不够多？ [GPU Occupancy](<GPU%20Occupancy.md>)

## Shared memory 访问方式好不好？[Bank Conflict](<Bank%20Conflict.md>)

## HBM /  global memory 访问方式好不好？[Memory Coalescing](<Memory%20Coalescing.md>)
## GPU 的瓶颈模型 [GPU Bottleneck](<GPU%20Bottleneck.md>)，这个就是 GPU 的 [GPU Memory Bound](<GPU%20Memory%20Bound.md>)
