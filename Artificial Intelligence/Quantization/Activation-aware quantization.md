#AI #LanguageModeling

[Lin+ 2023](https://arxiv.org/abs/2306.00978)

- Observation: some activation channels are large
- Weights that hit those matter more
- Allocate more precision to those weights
- Idea: select which weights (0.1-1%) to keep in high precision based on activations
	- AWQ 实际避免 mixed precision，通过 activation-aware 的 per-channel scaling 保护显著通道
- fp16 → int3 produces 4x lower memory, 3.2x speedup
![awq-schema](<../attachments/awq-schema.png>)
