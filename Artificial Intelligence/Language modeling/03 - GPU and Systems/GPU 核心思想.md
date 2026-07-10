#AI #LanguageModeling #GPU 

>[!note] CPU 解决 memory 很慢的方式
>我能不能让内存变快？
>于是出现：cache, branch prediction, prefetch
>这些解决 memory 很慢的手段

#### 但是GPU的思路：既然内存慢，那我就别等。
>[!example] 
>现在有
>```plain text
>Warp A
>Warp B
>Warp C
>Warp D
>```
>如果这时 Warp A load memory 的时候卡住了，GPU立刻切 Warp B 继续干，如果 Warp B 卡住了，就切 Warp C 继续干。等绕一圈回来，Warp A的数据已经回来了。

所以 GPU 的核心思想：**==Hide latency by parallelism.==** 而不是：Reduce latency.

Deep Learning 本质是大规模矩阵运算，天然拥有巨大并行度，所以特别适合 GPU。

GPU 设计的核心思维就是
- 怎么少搬数据
- 怎么让 SM 一直有活干