#AI #LanguageModeling #GPU

![memory overhead issue of DP.png](<../../attachments/memory%20overhead%20issue%20of%20DP.png>)

[Data parallelism](<./Data%20parallelism.md>) 中 Memory overhead 的问题，主要集中在 [Optimizer](<../../Transformer/Optimizer.md>) states 占用了大部分显存

>**Important** — Core idea
>**Split up the expensive states，并利用 [All-reduce Decomposition](<./All-reduce%20Decomposition.md>) 在 replicated 与 sharded results 之间转换。**

ZeRO 分为3个stages
```
ZeRO-1：切 optimizer states
ZeRO-2：再切 gradients
ZeRO-3 / FSDP：再切 parameters
```

和 Naïve [Data parallelism](<./Data%20parallelism.md>) 放在一起，这几个 memory 可以极简地写成：

```math
 \begin{aligned} \text{Naïve DP} &: P+G+O \\ \text{ZeRO-1} &: P+G+\frac{O}{M} \\ \text{ZeRO-2} &: P+\frac{G}{M}+\frac{O}{M}\\ \text{ZeRO-3} &:\frac{P}{M} +\frac{G}{M}+\frac{O}{M}\end{aligned}
```

而 Stage 2 并没有新增 collective：

---
## ZeRO Stage 1. Optimizer state sharding: $P_{OS}$

- Split up the optimizer state (first + second moments) across GPUs
- Everyone has the parameters + gradients
#### How it works

**Step 1.**  Everyone has the parameters + gradients
**Step 2.** 对 gradients 做 [Reduce-Scatter](<./Reduce-Scatter.md>)，让每个 ranks 拿到自己算 optimizer state 需要的那份 gradients
**Step 3.** Each machine updates their param using their gradient + state.
**Step 4.** [All-Gather](<./All-Gather.md>) parameters
#### Compare to [Data parallelism](<./Data%20parallelism.md>) its a free memory saving

设完整 gradient/parameter tensor 的大小都是 \(P\)。

Naïve DP：
```
Reduce-Scatter gradients       ≈ P
All-Gather gradients           ≈ P
-------------------------------
All-Reduce gradients           ≈ 2P
然后每个 rank 更新全部 parameters
```

ZeRO-1：

```
Reduce-Scatter gradients       ≈ P
每个 rank 只更新自己负责的 parameter shard
All-Gather updated parameters  ≈ P
-------------------------------
总通信量                        ≈ 2P
```

>**Important** — 相对于 naïve DP，没有增加通信量，却节省了 optimizer-state memory。

---
## ZeRO Stage 2. the simple extension to gradient sharding: $P_{OS+g}$

- Also keep the gradients (pink slices) sharded across the machines.
- Use the same (rough) tricks as stage 1.

每个 rank 仍然：
- 有完整 parameters；
- 用自己的 local batch 做完整 [forward pass](<../../Neural%20Networks/Forward%20Propagation.md>) / [backward pass](<../../Neural%20Networks/Backpropagation.md>)；
- 计算所有 parameters 的 local gradients。
区别在于什么时候通信和释放。

#### How it works

**Step 1.** 每个 rank layer by layer 执行 backward。
	**Step 1a.** 某层的 local gradients 一生成，就立即进行 [Reduce-Scatter](<./Reduce-Scatter.md>)。
	![Reduce.png](<../../attachments/Reduce.png>)
	**Step 1b.** Reduce-Scatter 完成后，释放该层未分片的 local gradient buffer；每个 rank 只长期保留自己负责的 global gradient shard。
**Step 2.** 每个 rank 使用自己负责的 gradient shard 和 optimizer state，更新对应的 parameter shard。
**Step 3.** [All-Gather](<./All-Gather.md>) parameters

最后没有任何 rank 保存完整 gradient：

```
rank 0：保存 global gradient shard 0
rank 1：保存 global gradient shard 1
rank 2：保存 global gradient shard 2
...
```

#### Communication Cost

Stage 2 并没有新增 collective：

```
Reduce-Scatter gradients
→ shard update
→ All-Gather updated parameters
```

它只是把 Reduce-Scatter 提前到 backward 过程中，一边产生 gradients，一边通信并释放。因此通信量和 Stage 1 基本相同，又额外省下了 gradient memory。

```
总通信量                        ≈ 2P
```
---
## ZeRO Stage 3. (aka FSDP) shard everything: $P_{OS+g+p}$

Compared to stage 2 shard parameters

>**Note** — 完整的模型不长期储存在任何一张 GPU 上；但是当计算到某一层时，完整的这一层会临时出现在每张 GPU 上

| Stage  | 每张 GPU 长期保存                                                   |
| ------ | ------------------------------------------------------------- |
| ZeRO-2 | 完整 parameters，\(1/M\) gradients，\(1/M\) optimizer states      |
| ZeRO-3 | \(1/M\) parameters，\(1/M\) gradients，\(1/M\) optimizer states |

>**Question** — 如果每张 GPU 只有一部份 weights，它怎么完成这一层的 forward?
>Send and request parameters on demand while stepping through the compute graph. 计算前临时 [All-Gather](<./All-Gather.md>)
>

#### How it works? （用一个简单的 Linear 来举例）

假设一层是：
```math
Y=XW
```
两张 GPU 分别长期保存：
```
rank 0：W 的 shard 0
rank 1：W 的 shard 1
```
但是 rank 0 和 rank 1 各自有自己的 local batch，所以都需要完整的 $W$ 来计算 $XW$

###### Forward
```
1. All-Gather W
   每个 rank 临时得到完整 W

2. 每个 rank 用自己的 local X 计算 Y = XW

3. forward 完成
   释放其他 ranks 的 W shards
   重新只保留自己的 W shard
```
⚠️：forward 中 backward 所需的一部分 [activations](<../../Neural%20Networks/Activations.md>) 仍需暂时保存，除非使用 [Recomputation](<../02%20-%20Training%20and%20Scaling/Recomputation.md>)；这里释放的是临时 All-Gather 出来的完整 weights。

###### Backward
Backward 又需要 $W$，例如：
```math
\nabla_XL=\nabla_YLW^\top
```
```
1. 再次 All-Gather W
   临时重建当前层的完整 weights

2. 每个 rank 用自己的 local batch 计算 local gradients

3. Reduce-Scatter gradients
   聚合各 rank 的 local gradients
   每个 rank 只保留自己负责的 global gradient shard

4. 释放临时的完整 W
```

##### Tradeoff in stage 3

多通信一次，换取 parameters 始终只以 shard 的形式长期存在。

 #### Cost

从上面的例子可以看出来，Communication Cost - 2 $\times$ [All-Gather](<./All-Gather.md>)，1 $\times$ [Reduce-Scatter](<./Reduce-Scatter.md>)

```
Forward parameters All-Gather   ≈ P
Backward parameters All-Gather  ≈ P
Gradient Reduce-Scatter         ≈ P
-----------------------------------
总通信量                         ≈ 3P
```

但是这个 cost 不绝对，因为它多了一个 parameters All-Gather，但是这个 All-Gather 的 Communication 和 Computation overlap

比如：

```
正在计算 layer 0
同时预取、All-Gather layer 1 的 weights

正在计算 layer 1
同时预取、All-Gather layer 2 的 weights
```
