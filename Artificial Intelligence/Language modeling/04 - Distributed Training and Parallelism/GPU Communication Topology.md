
![gpu-node-overview.png](<../attachments/gpu-node-overview.png>)

多 GPU 训练可以理解成把 [GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>) 的 memory hierarchy 往外扩展：

1. **单张 GPU 内部**：SM 从 registers / shared memory / HBM 中取数据，重点是减少 HBM ↔ compute 的搬运。
2. **单机多 GPU**：多张 GPU 通过 **NVLink / NVSwitch** 通信，通常 bandwidth 较高，适合 communication-heavy 的 parallelism。
3. **多机多 GPU**：不同 node 之间通过 **InfiniBand / Ethernet / RoCE** 通信，距离更远、通信更贵。

所以 [Parallelism](<./Parallelism.md>) 的问题不是只看“能不能切”，而是要看切完之后 ranks 之间需要通信多少、通信路径有多快。

[NCCL](<./NVIDIA%20Collective%20Communication%20Library.md>) 负责把 collective operations（例如 all-reduce、all-gather、reduce-scatter）映射到底层 GPU 通信拓扑上，让多张 GPU 能高效交换 tensor。

>**Note** — 和单卡 GPU 优化的联系
>单卡优化是在减少 HBM ↔ compute 的数据搬运；多卡 parallelism 是在减少和组织 GPU ↔ GPU / node ↔ node 之间的数据通信。

---
# 🔗
[GPU](<../03%20-%20GPU%20and%20Systems/GPU.md>)
[Parallelism](<./Parallelism.md>)
[NVIDIA Collective Communication Library](<./NVIDIA%20Collective%20Communication%20Library.md>)
