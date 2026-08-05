#AI #LanguageModeling

[Frantar+ 2022](https://arxiv.org/abs/2210.17323)

先正常训练模型：BF16 / FP16 / FP32 model
训练完之后再压缩：INT8 / FP8 / INT4

>**优点**：简单不用重新训练太久部署友好
>**缺点**：精度可能掉尤其低到 4-bit / 3-bit 时更明显


- Done after training, so much cheaper
- Run on sample data to determine scale and zero point for each layer or tensor
- GPTQ: use Hessian information to update non-quantized weights to account for quantization error
