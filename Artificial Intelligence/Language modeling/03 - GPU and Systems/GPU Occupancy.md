#AI #LanguageModeling #GPU

这是衡量 [GPU](<./GPU.md>) performance 的一个指标

>**Question** — 一个 SM 资源有限。每个 thread 都要用一些 registers。但是如果每个 thread 用太多的 registers，那一个 SM 同时能运行的 threads / warps 就变少

假设：
- 一个 thread block 有 128 个 threads
- 一个 warp 是 32 个 threads
- 每个 thread 要用 160 个 registers
- SM 一共有 65536 个 registers
- 硬件最多支持 64 个 warps 同时挂在一个 SM 上

occupancy 计算如下：
- 一个 block 总共要用 $128\times 160=20480$ 个 registers
- 所以这个 SM 最多同时能放 $65536 / 20480=3$ 个这样的 blocks
- 一个 block 有 $128/32=4$ 个 warps
- 所以 SM 上有 $3\times 4=12$ 个 warps
- 所以 occupancy = $12 /64=18.75\%$

>理论上 SM 可以挂 64 个 warps，但因为每个 thread 用 register 太多，实际只能挂 12 个 warps，所以 occupancy 很低。

## occupancy 衡量的什么

GPU 靠“多挂很多 warp”来隐藏 memory latency
- occupancy 高 -> SM 手里有很多 warps 可以切换，不会闲着
- occupancy 低 -> 只有很少的 warp，一旦它们都在等 HBM，SM 就没活干

所以 occupancy 衡量的是：一个 SM 上有多少可调度的 warps 可以来填满时间。但是这个也不是越高越好，因为 thread 用的 registers 多，它的单位效率会高一些

>**Summary** — My understanding
>Occupancy 关心一个 SM 上同时有多少可调度的 warps。它受 registers、shared memory、thread block size 等资源限制。
