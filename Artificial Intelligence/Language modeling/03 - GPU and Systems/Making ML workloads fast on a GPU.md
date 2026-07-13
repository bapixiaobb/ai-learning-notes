#AI #MachineLearning #LLM #Architecture #LanguageModeling #GPU

>**Question** — 一个模型的计算结构会不会适合 GPU / TPU 这种硬件？
> 比如 Transformer 适合现代 [GPU](<./GPU.md>)，是因为它大量使用：**Matrix Multiplication**
>例如：
>```
>XW
>QK^T
>Attention V
>MLP projection
>```
>这些都可以很好地喂给 Tensor Core / matrix multiply unit。

现代 [Machine Learning](<../../Fundamentals/Machine%20Learning.md>) Model Architecture 会倾向于设计成大量矩阵乘法，因为硬件最擅长矩阵乘法。  
也就是：**模型架构会被硬件形状反向塑造。** ⚙️

（现代 deep learning 模型架构会被硬件影响。因为 GPU/TPU 最擅长 matrix multiplication，所以成功的大模型架构通常会包含大量 matrix multiplication。）

# 🔗
[Large Language Model (LLM)](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>)
[Language Modeling](<../00%20-%20Maps%20and%20Overview/Language%20Modeling.md>)
[Model Architecture](<../05%20-%20Architectures%20and%20MoE/Model%20Architecture.md>)