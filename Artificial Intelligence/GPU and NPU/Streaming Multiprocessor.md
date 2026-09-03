#GPU

[GPU](<GPU.md>) 真正工作的地方
## 硬件结构：SM 里面有什么

每个 SM 里有

```
SM
├── Warp Scheduler / Dispatch
├── Register File
├── Shared Memory
└── Execution Units
    ├── CUDA Cores：普通 FP/INT 算术
    ├── Tensor Cores：矩阵 MMA
    ├── Load/Store Units：访存
    └── Special Function Units：exp、sin 等特殊函数
```

>**Note** — SM负责：执行Warp，切换Warp，隐藏Latency

- **ALU**：算术逻辑单元的泛称。
- **[CUDA Core](<CUDA%20Core.md>)**：NVIDIA 对普通算术执行单元的命名，主要执行 `add`、`mul`、`fma` 等标量指令。
- **[Tensor Core](<Matmul/Tensor%20Core.md>)**：执行矩阵 Tile 级别的 `D=A×B+C`。

## What is Warp and Why we need it?
GPU 为了省调度成本，会把32个 Thread 绑在一起，变成 Warp（32 人施工队），它们必须同时执行同一条指令，例如：一起做 $\times$，或者一起做 $+$

这个叫：**[SIMT](<SIMT.md>)** (Single Instruction Multiple Threads) -- 这是 GPU 的 Execution Model

GPU 一次性需要管理 10000 个线程，太贵了。所以从硬件执行角度，SM 的 scheduler 调度的是 warp。Scheduler 调度的是 Warp 不是 Thread
