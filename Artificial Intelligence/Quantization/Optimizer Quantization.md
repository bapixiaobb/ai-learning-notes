#AI #LanguageModeling #Quantization

在 [Optimizer](<../Transformer/Optimizer.md>) quantization 中，主要的量化对象是 [AdamW](<../Transformer/AdamW.md>) 长期保存的 first moment ($m$) 和 second moment ($v$)。二者与 parameters shape 相同，通常以 FP32 保存，并会跨越整个 training run 持续累积，因此是 optimizer memory 的主要来源。

Low-bit optimizer 通常只降低 ($m,v$) 的存储精度：在 optimizer step 中先将它们 dequantize 到 FP32，完成 accumulator 和 parameter update，再重新 quantize 保存。FP32 moments 仍然是稳健默认；[quantization](<Quantization.md>) 与 [ZeRO](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/ZeRO.md>) / FSDP sharding 是可以组合的两种 memory optimization。

![Optimizer memory breakdown](<../attachments/max%20memory.png>)

**完整生命周期**

| Tensor                    | 保存多久                      | 主要作用                   | Quantization 价值                              |
| ------------------------- | ------------------------- | ---------------------- | -------------------------------------------- |
| Model parameters $\theta$ | 整个 training run           | forward/backward       | persistent memory、带宽、计算                      |
| FP32 master weights       | 整个 training run，可选        | 保存高精度 parameters       | persistent memory                            |
| AdamW $m,v$               | 整个 training run           | 历史 gradient statistics | persistent memory（optimizer quantization 重点） |
| Gradients                 | backward 到 optimizer step | 当前 update              | peak memory、通信                               |
| Activations               | forward 到对应 backward      | backward 计算 gradient   | peak memory、计算和带宽                            |
| Temporary buffers         | 某个 operator/step          | 中间计算                   | peak memory                                  |

## Master weight $\theta^{FP32}$

是由 mixed-precision optimizer 创建和维护，为了保存 optimizer 产生的微小 parameter updates

流程：

```math
\theta_t^{\mathrm{BF16}} \xrightarrow{\text{forward/backward}} g_t
```

Optimizer 把 gradient 转到适合计算的精度，更新 FP32 moments：

```math
m_t,v_t \leftarrow \operatorname{AdamStateUpdate}(g_t)
```

然后得到 FP32 parameter update：

```math
\Delta\theta_t = -\alpha_t \frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
```

这个 update 加在 FP32 master weights 上：

```math
\theta_{t+1}^{\mathrm{master}} = \theta_t^{\mathrm{master}}+\Delta\theta_t
```

最后再生成 BF16 model parameters：

```math
\theta_{t+1}^{\mathrm{BF16}} = \operatorname{cast} \left(\theta_{t+1}^{\mathrm{master}}\right)
```

## Gradient

optimizer 中的 gradient 是 current-step state，通常不属于 persistent optimizer state。

可能占据大量峰值显存，而且 [data parallel](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/Data%20parallelism.md>) 需要通信 gradients，因此它的量化研究重点通常是：

- backward memory；
- All-Reduce communication；
- low-precision training。

## Optimizer state

是 optimizer quantization 的重心。**当前比较成熟、使用较多的是 8-bit Adam/[AdamW](<../Transformer/AdamW.md>) optimizer states。**  
4-bit 和 FP8 optimizer states 已经有实现，但更多处于逐步落地阶段。大规模预训练仍常见 FP32 moments 配合 [ZeRO](<../Language%20modeling/04%20-%20Distributed%20Training%20and%20Parallelism/ZeRO.md>)/FSDP。

Optimizer state 通常量化的是 **persistent $m,v$ 的存储**：

```math
\text{low-bit storage} \rightarrow \text{dequantize to FP32} \rightarrow \text{update }m,v,\theta \rightarrow \text{requantize}
```

8-bit optimizer 论文明确采用这种方式：state 以 8-bit 保存，在 registers 中反量化并完成 FP32 update，再压回 8-bit。[8-bit Optimizers](https://arxiv.org/abs/2110.02861)

这套方法的官方工程实现之一是 [bitsandbytes](https://huggingface.co/docs/bitsandbytes/optimizers) 提供的 `AdamW8bit`：它将 persistent $m,v$ 以 block-wise 8-bit 形式保存，在 `optimizer.step()` 中临时反量化到 FP32 完成更新，再将新的 states 压回 8-bit。

#### Block-wise 8-bit AdamW Recipe

**Ingredients：** AdamW 的 $m,v$、每个 block 的 FP32 scale、signed / unsigned 8-bit codebook。

```math
(q_m,q_v,\text{scales})
\rightarrow \text{FP32 AdamW update}
\rightarrow (q_m',q_v',\text{new scales})
```

>**Note**
> 当前实用主线是 block-wise `AdamW8bit`；`PagedAdamW8bit` 更偏 memory-constrained fine-tuning。4-bit / FP8 optimizer states 已经出现，但还不是默认 training recipe。

See: [bitsandbytes AdamW API](https://huggingface.co/docs/bitsandbytes/reference/optim/adamw) and [8-bit optimizer explanation](https://huggingface.co/docs/bitsandbytes/explanations/optimizers).
