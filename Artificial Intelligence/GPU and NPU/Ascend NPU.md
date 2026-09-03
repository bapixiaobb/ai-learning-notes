#AI

Neural Processing Unit，Ascend NPU 是华为面向[神经网络](<../Neural%20Networks/Neural%20Network.md>)计算设计的处理器。

神经网络的主要工作可以粗略分为：
- 矩阵计算
- 向量计算
- 数据搬运
- 控制与地址计算
Ascend 的核心思路是：**既然这些工作性质不同，AI Core 就将这些工作交给不同的硬件单元，并让它们尽可能同时工作。**
- **计算单元**：包括Cube（矩阵）计算单元、Vector（矢量）计算单元和Scalar（标量）计算单元。
- **存储单元**：包括L1 Buffer、L0A Buffer、L0B Buffer、L0C Buffer、Unified Buffer、BiasTable Buffer、FixPipe Buffer等专为高效计算设计的存储单元。
- **搬运单元**：包括MTE1、MTE2、MTE3和FixPipe，用于数据在不同存储单元之间的高效传输。

## 传统 NPU：从张量分块和流水线出发

一次典型计算的行为链是：
```text
大 Tensor 位于 Global Memory
    → 切出当前 Tile
    → Tile 从 GM 搬到片上 Local Memory
    → Cube / Vector 流水计算
    → 结果写回 Global Memory
```
这种范式对规则、密集计算非常高效，但对随机索引、细粒度分支和线程级控制不够自然。

>**Note** — 这里的 Local Memory 是片上 Buffer 的总称，主要包括：
>- Vector 使用的 UB。
>- Cube 使用的 L1、L0A、L0B、L0C。
>
>A5 的片上存储与数据路径还增加了 SIMD Register、SIMT Register、Data Cache 等结构；它们不应全部简单等同于传统 Local Memory Buffer。

Tiling 对不同算子的收益不完全相同：
- 对 Add 等逐元素算子，每个元素仍然必须从 GM 读入并写回，Tiling 主要用于匹配硬件搬运粒度，并支持不同 Tile 之间形成流水。
- 对 MatMul 等存在数据复用的算子，A/B Tile 搬入片上后可以参与多次乘加，因此还能减少重复 GM 访问并提高 Arithmetic Intensity。

#### Ascend 这里可以有两层并行：
1. **核间并行**
	- 一个很大的 Tensor 会先被切分，不同计算核心负责不同的数据块。
		- 这和 [GPU](<GPU.md>) 上多个 SM 同时处理不同 Blocks 的目的相似，但两者的调度与编程抽象并不相同。
	- 多个 Core 同时处理不同 Tile
2. **核内并行**
	- 搬运流水与当前 Core 对应的计算流水尽量重叠。
	- Cube 与 Vector 是否位于同一个 Core、如何协同，取决于具体芯片采用耦合架构还是分离架构。

---
## 流水线

### Vector

例如 Add、Exp、比较、部分归约：
```
GM → UB → Vector → UB → GM
```
1. 输入 Tile 从 GM 搬入 UB。
2. Vector 对 UB 中的数据进行计算。
3. 结果仍写入 UB。
4. 最后从 UB 搬回 GM。

A5 的 SIMD 路径中间又增加了 Vector Register：
```
GM → UB → RegTensor → Vector → RegTensor → UB → GM
```

### Cube

矩阵乘：
```
GM → L1 → L0A/L0B → Cube → L0C → FixPipe → GM/L1
```
- L1：暂存较大的 A/B Tile。
- L0A：保存 Cube 当前使用的左矩阵小块。
- L0B：保存右矩阵小块。
- Cube：执行矩阵乘加。
- L0C：保存并累加结果。
- FixPipe：把 L0C 结果转换或搬运出去。

---
## Latency

[GPU 以 Warp 等待长延迟内存访问为典型场景，所以用更多就绪 Warp 来 hide latency](<GPU%20Performance.md#execution-model>)。数据依赖和某些执行流水也可能使 Warp 暂停。

传统 Ascend 更强调：
```
搬入 Tile 1
计算 Tile 0
搬出 Tile -1
```
三件事分别由不同流水处理，因此可以重叠。

这种重叠发生在**不同 Tile、不同硬件流水**之间；对于同一个 Tile，`CopyIn → Compute → CopyOut` 的数据依赖仍然必须满足。它也不表示同一个 Vector 或 Cube 单元能同时计算多个 Tile。

Double Buffer 的作用是在同一片 UB 空间中分配两块互不重叠的工作区，而不是增加两块物理 UB：
```
Vector 从 Buffer 0 读取并计算 Tile 0
MTE 同时向 Buffer 1 搬入 Tile 1
Tile 0 使用完毕后，两块 Buffer 交换角色
```

如果只有一个 Buffer，MTE 提前写入 Tile 1 就可能覆盖 Vector 尚未读完的 Tile 0，导致数据竞争和错误结果。Double Buffer 通过读写不同的 Buffer 避免这一冲突，同时让搬运和计算得以重叠。

>**Important** — 所以传统 Ascend 的核心性能思路是：
> **通过 Tiling、片上复用、Double Buffer 和多流水并行，让搬运单元与计算单元同时工作。**

这和 GPU 的最终目标相同——让计算单元不要等数据；主要机制有所不同。
