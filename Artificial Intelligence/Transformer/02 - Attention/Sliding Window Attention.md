#AI #LanguageModeling

一种 [Reduce KV cache size](<../../Language%20modeling/06%20-%20Inference%20and%20Serving/Reduce%20KV%20cache%20size.md>) 的方法

![longformer-attention](<../../attachments/longformer-attention.png>)

Idea: just look at the local context, which is most relevant for modeling

Effective context scales linearly with the number of layers

KV cache is independent of sequence length!

Problem: this can still hurt accuracy
Solution: interleave local attention with global attention (hybrid layers)
