#AI #LLM #LanguageModeling

# Menu

**把高精度数映射到低精度表示**

比较典型的例子是
```
FP32 / BF16 tensor
↓
quantize
↓
INT8 / FP8 / INT4 tensor + scale
```

然后需要时：

```
low precision tensor + scale
↓
dequantize
↓
approx FP32 / BF16 tensor
```

# Quantization $\neq$ [Low precision computation](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Low%20precision%20computation.md>)

>**Note** — Quantized Storage
>比如有些部署流程会这样：
>```
>INT8 weight 存在硬盘 / 显存里
>↓
>运行时 dequantize 成 FP16 / BF16
>↓
>再用 FP16 / BF16 matmul
>```
>这里发生了 quantization，因为存储时是 INT8。但真正计算时可能不是 INT8 compute，而是先还原到 BF16 再算。所以这种情况不算是 [Low precision computation](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Low%20precision%20computation.md>)

---
# State of the Art

### Post-training quantization

先正常训练模型：BF16 / FP16 / FP32 model
训练完之后再压缩：INT8 / FP8 / INT4
>**优点**：简单不用重新训练太久部署友好
>**缺点**：精度可能掉尤其低到 4-bit / 3-bit 时更明显

### Quantization-aware training

训练时就考虑量化误差。
比如训练过程中模拟：如果这个 weight 未来会被量化，它现在应该怎么学？
这样模型会适应低精度。
>**优点**：质量通常更好
>**缺点**：训练更复杂成本更高

### Train big, quantize down

这是现在很常见的实际路线：先训练大模型 / 高精度模型↓再量化成部署模型
比如为了在本地电脑、手机、边缘设备上跑。

>**Important** — Matmul 做 quantization 收益最大
> 硬件有专门的适配 FP8 / MXFP8 的 tensor core，做 [Low precision computation](<../Language%20modeling/02%20-%20Training%20and%20Scaling/Low%20precision%20computation.md>)
