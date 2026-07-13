#DeepLearning #Systems

>**Note**
> **Deep Learning Systems** 研究如何高效训练和部署 deep learning models，重点关注 computation、memory、communication 和 hardware utilization。

## 核心问题

>**Question**
> 给定模型、数据和硬件，如何让训练或推理更快、更省显存、更高效？

## 主要内容

- [Resource Accounting](<../02%20-%20Training%20and%20Scaling/Resource%20Accounting.md>)
- [FLOPs](<./FLOPs.md>)
- FLOP per Second
- Memory Bandwidth
- Roofline Model
- [Model FLOPs Utilization](<./Model%20FLOPs%20Utilization.md>)
- Mixed Precision
- Parallelism

## 与 Large Language Model 的关系

>**Note**
> 现代 [Large Language Model (LLM)](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>) 训练成本极高，因此需要 deep learning systems 方法来估算和优化 compute、memory、communication 和 training throughput。

## Related

- [Deep Learning](<../../Neural%20Networks/Deep%20Learning.md>)
- [Large Language Model (LLM)](<../00%20-%20Maps%20and%20Overview/Large%20Language%20Model%20(LLM).md>)
- [Transformer](<../../Transformer/Transformer.md>)
- An Optimization Problem
- Linear System