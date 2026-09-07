#AI #LanguageModeling #Quantization

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

# Quantization $\neq$ [Low precision computation](<../Language%20modeling/03%20-%20Training%20and%20Scaling/Low%20precision%20computation.md>)

> **Note** — Quantized Storage
>比如有些部署流程会这样：
>```
>INT8 weight 存在硬盘 / 显存里
>↓
>运行时 dequantize 成 FP16 / BF16
>↓
>再用 FP16 / BF16 matmul
>```
>这里发生了 quantization，因为存储时是 INT8。但真正计算时可能不是 INT8 compute，而是先还原到 BF16 再算。所以这种情况不算是 [Low precision computation](<../Language%20modeling/03%20-%20Training%20and%20Scaling/Low%20precision%20computation.md>)

---
# State of the Art

### [PTQ](<Post-training%20quantization.md>)

### [QAT](<Quantization-aware%20training.md>)

### [AWQ](<Activation-aware%20quantization.md>)

### Train big, quantize down

这是现在很常见的实际路线：先训练大模型 / 高精度模型↓再量化成部署模型
比如为了在本地电脑、手机、边缘设备上跑。

> **Important** — Matmul 做 quantization 收益最大
> 硬件有专门的适配 FP8 / MXFP8 的 tensor core，做 [Low precision computation](<../Language%20modeling/03%20-%20Training%20and%20Scaling/Low%20precision%20computation.md>)
