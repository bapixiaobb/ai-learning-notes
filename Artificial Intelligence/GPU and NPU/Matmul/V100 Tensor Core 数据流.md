#GPU


```
GMEM
  │ 普通 global load
  ▼
Thread Registers          ← 第一次经过寄存器
  │ store
  ▼
Shared Memory
  │ wmma.load / matrix load
  ▼
Thread Registers          ← A/B register fragments
  │ warp-level MMA
  ▼
Tensor Core
  │
  ▼
Thread Registers          ← accumulator fragments
  │ 后处理、保存
  ▼
SMEM / GMEM
```

🙋每份数据都需要：从 GMEM 加载到 Thread Register；再从 Thread Registers 写入 Shared Memory
## 寄存器压力

一个进行矩阵计算的 Warp，需要同时保存：

```
A fragment
B fragment
accumulator fragment
地址、循环变量
其他临时数据
```

如果一个 Thread 使用的寄存器很多，那么一个 SM 能同时容纳的 Warp 就会减少：

```
每个 Thread 使用更多 registers
        ↓
一个 Warp 使用更多 registers
        ↓
一个 SM 能驻留的 Warp 数量下降
        ↓
可用于隐藏 latency 的 Warp 变少
```

所以 Volta 的矛盾是：

> Tensor Core 算矩阵很快，但喂给它的数据和它产生的结果仍然大量依赖普通 Thread Register File。
