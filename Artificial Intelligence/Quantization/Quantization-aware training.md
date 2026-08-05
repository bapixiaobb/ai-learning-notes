#AI #LanguageModeling

训练时就考虑量化误差。
比如训练过程中模拟：如果这个 weight 未来会被量化，它现在应该怎么学？
这样模型会适应低精度。

>**优点**：质量通常更好
>**缺点**：训练更复杂成本更高


- During training, quantize-and-dequantize during the forward pass to simulate quantization errors
- Pro: weights are trained to work with quantization
- Con: requires expensive large-scale training
