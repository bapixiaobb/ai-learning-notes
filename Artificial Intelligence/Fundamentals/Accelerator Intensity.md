#AI #GPU

硬件指标

```math
I_{\text{accel}}
=
\frac{\text{peak FLOP/s}}{\text{memory bandwidth}}
```

## Example

**Accelerator intensity of H100 SXM for dense FP16/BF16 Tensor Core compute:**
```
peak_flops = 989e12                         # dense FP16/BF16 Tensor Core FLOP/s
memory_bandwidth = 3.35e12                  # HBM bytes/s

accelerator_intensity = peak_flops / memory_bandwidth
accelerator_intensity ≈ 295 FLOP/byte
```

在假设 dense FP16/BF16 Tensor Core 计算时，算子的 arithmetic intensity 大约需要超过 `295 FLOP/byte`，理论上才可能从 memory-bound 转向 compute-bound。
