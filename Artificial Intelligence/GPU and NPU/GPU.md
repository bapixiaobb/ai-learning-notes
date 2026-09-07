#AI #LanguageModeling #GPU
# GPU hardware Anatomy

>**Note**
> GPU 是一个 throughput machine。它不是为了让一个任务最快完成，而是为了让大量相似任务并行完成。

```
GPU
 ├── SM
 │    ├── registers
 │    ├── shared memory
 │    ├── warp scheduler
 │    ├── ALU / CUDA cores
 │    └── tensor cores
 └── HBM
```
### Compute Side

CPU 相当于有 8 个博士，每个人都很聪明
GPU 相当于有 100 个小学生，每个人只会 $+$ 和 $\times$，但矩阵乘法刚好就是：$+$ 和 $\times$
####  🏭 What is [SM](<Streaming%20Multiprocessor.md>)?
GPU 不是10000个核心直接平铺，更像：
```
GPU
 ├─ 工厂1
 ├─ 工厂2
 ├─ 工厂3
 ├─ ...
 └─ 工厂N
```
这些工厂就是：SM

这是GPU真正工作的地方。可以理解成：一个车间。

>**Example** — H100 大约有 132 个 SM

### Memory Side

>**Note** — 硬件其实关心的主要是
>A 在哪？B 在哪？
>

```
HBM
 ↓
Shared Memory
 ↓
Register
 ↓
Compute Units
```
对于 GPU 来说，数据的位置比计算本身更重要。GPU 慢很多情况下，不是因为算力不够，而是数据搬运太慢。

GPU 里面有很多层存储：
#### HBM / global memory（总仓库）

HBM = High Bandwidth Memory，就是 GPU 显存
例如：
- H100 80GB
- A100 80GB
这些80GB基本都是 HBM。

特点：✅ 容量巨大；❌ 延迟高；❌ 离计算单元远

#### Shared Memory / SRAM （SM内部的小仓库）

一般只有几十KB

特点：✅ 比HBM快很多；❌ 很小

#### Register （👷手里的工具）

这是最快的储存。一个线程可能只有几十个寄存器

特点：✅ 极快；❌ 极小

>**Note** — 现代 GPU 矛盾
>```
>计算增长速度 >>> 带宽增长速度
>```
>算的太快，数据拿的太慢

>**Important** — GPU 是一个计算能力远远大于内存带宽的机器。

---
# GPU Programming Model
>**Note**
> Programming model 是程序员组织任务的方式，不是 GPU 里面真实长出来的硬件零件。

```
kernel launch
 └── grid
      └── thread blocks / CTAs
           └── threads
```

kernel launch 发起一次 GPU 任务
grid 包含任务里所有的 blocks (thread blocks)
blocks / CTA 是一组能协作的 threads
[Thread](<Thread.md>) 是最小编程单位，一个线程处理一小份数据

#### How gird /  block / thread work on GPU? --- by SM
所以这里的工作流程是：一个 thread block 是一个软件任务小队，会被派到某个 SM 车间里执行

```
thread block 被放到某个 SM 上
thread block 里的 threads 会被硬件分成 warps
SM 里的 warp scheduler 负责调度这些 warps
每个 thread 用自己的 registers
同一个 block 内的 threads 可以共享这个 SM 上的 shared memory
```

---
# Mapping: From Program to Hardware

```
grid 里的很多 thread blocks
        ↓ 被调度
GPU 上的很多 SM
        ↓ 执行
block 里的 threads
        ↓ 硬件上按 warp 分组执行
warp scheduler 调度 warps
```

---
## Parallelism

多 GPU 训练可以看成把 GPU 的 memory hierarchy 往外扩展了一层，具体的硬件连接方式见 [GPU Communication Topology](<GPU%20Communication%20Topology.md>)。

单张 GPU 可能放不下完整的 model states（parameters / gradients / optimizer states）和 [activations](<../Neural%20Networks/Activations.md>)，或者即使放得下，也希望用更多 GPU 提高训练吞吐量。因此需要把训练任务切到多个 GPU / ranks 上。

首先要决定 [Parallelism](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/Parallelism.md>)，也就是怎么切：切 data、切 tensor、还是切 layers。
一旦切开，不同 ranks 手里就只有局部信息，因此需要 collective communication 来交换、聚合或重建缺失的信息。

[NCCL](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/NVIDIA%20Collective%20Communication%20Library.md>) 负责把这些 collective operations（如 all-reduce、all-gather、reduce-scatter）高效地映射到底层 GPU 通信硬件上。

>**Note** — 单卡 GPU 优化是在减少 HBM ↔ compute 的数据搬运；多卡 parallelism 则是在减少和组织 GPU ↔ GPU 之间的数据通信。


---
# 🔗
[Language Modeling](<../Language%20modeling/00%20-%20Maps%20and%20Overview/Language%20Modeling.md>)
[GPU 核心思想](<GPU%20%E6%A0%B8%E5%BF%83%E6%80%9D%E6%83%B3.md>)
[GPU vs](<GPU%20vs.%20CPU.md>)
[Language Model Architecture](<../Language%20modeling/02%20-%20Language%20Modeling%20Basics/Language%20Model%20Architecture.md>)
[Making ML workloads fast on a GPU](<Making%20ML%20workloads%20fast%20on%20a%20GPU.md>)
