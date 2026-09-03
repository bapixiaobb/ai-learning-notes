#GPU

NVIDIA 对普通算术执行单元的命名，主要执行 `add`、`mul`、`fma` 等标量指令。

```
Warp Scheduler
      │
      │ 发射一个 warp 的 add/fma 指令
      ▼
CUDA Core 执行通道
      │
      └── 执行各个 active thread 对应的 scalar 运算
```

Warp 是调度和指令发射分组，CUDA Core 是执行运算的硬件。一个 warp 是否需要多个周期执行完，取决于具体 [GPU](<GPU.md>) 有多少执行通道
