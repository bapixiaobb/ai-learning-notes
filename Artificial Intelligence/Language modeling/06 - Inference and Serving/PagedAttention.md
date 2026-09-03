#AI #LanguageModeling

vLLM 提出的 [Kwon+ 2023](https://arxiv.org/pdf/2309.06180.pdf)，用来解决 [Inference](<Inference.md>) 中 [KV Cache](<KV%20Cache.md>) size 过大的问题，但是这个解决不是 ⬇️ 它的大小，而是改变 memory layout，让 KV 数据更紧凑、灵活地放进 [GPU](<../../GPU%20and%20NPU/GPU.md>) [memory](<../../GPU%20and%20NPU/GPU%20Memory%20Bound.md>)。

## KV Cache 的 memory allocation
![paged-attention-fragmentation](<../../attachments/paged-attention-fragmentation.png>)
👆连续的格子就相当于 GPU memory，图中每个 slot 概念上用来保存：某个 request 的一个 token 对应的 KV Cache state

Request A 到来时，服务器知道 prompt 有多长，但不知道模型最终会生成多少 tokens。

旧方法要求 KV cache 在物理 memory 中连续，为了保证后续 KV cache 能连续增长，会提前给 A 圈出一整段连续空间，就是上图的 our 之后的一直到 external fragmentation 这一段

可以看出深棕色的这一段（\<resv\>）和灰色这段
```math
\boxed{ \text{Internal fragmentation} = \text{allocation 内部预留但未使用的空间} }
```
```math
\boxed{ \text{External fragmentation} = \text{allocation 之间分散且难以利用的 free holes} }
```

## Paging

把 sequence 的 KV cache 分到 non-contiguous **blocks** 上
![paged-attention-blocks](<../../attachments/paged-attention-blocks.png>)
维护一张 block table，上面的 Block 0 就是 Logical Block（顺序的），所以👆虽然 Physical Block 乱序，request 仍然认为自己的 sequence 是：
```
Four → score → and → seven → years → ago → our → fathers → brought → forth
```

所以这种情况下，在 [Generation](<Generation.md>) 的过程中，不需要在一个满的 block 后面找空间，直接找一个 空闲的 Physical Block 就好

> **Important** — Attention
> 这个切分不只是切到 KV Cache，对于 [Attention](<../../Transformer/02%20-%20Attention/Multi-Head%20Attention.md>) 的 kernel 来说，也要求 K/V 是一段连续 tensor，所以 attention kernel 也需要理解这种 block mapping

## KV-block pool

对于这种方法来说，requests 可以共享 KV caches
![paged-attention-logical](<../../attachments/paged-attention-logical.png>)

Request A 和 B 各自维护 block table；即使它们共享同一片 physical memory，每个 request 也只看到自己的 logical address space

> **Note** — Logical grow
> Request A 左边还画了一个空的 Logical Block 3；Request B 右边也有空的 Logical Block 2
>
> 它们表示未来可能增长的位置，但当前并没有对应的 physical allocation。
>
> 也就是说：逻辑上允许继续增长 ≠ 现在就预留 physical memory

## Share prefix

这个想法比较直观
![paged-attention-parallel](<../../attachments/paged-attention-parallel.png>)

每个 request 有自己的逻辑地址空间和 block table，但不同 block tables 可以指向同一个 physical KV block。

#### 真正的冲突发生在写入时

如果共同 prefix 正好填满 block，👆 Block 1 和 Block 3

Copy-on-write：在真正发生不同写入时，为其中一个 request 复制一份 block，然后各自修改。
