#AI #LanguageModeling #GPU

>[!note]
> **Memory Coalescing** 发生在 HBM / global memory 访问中。
>
> 当同一个 warp 里的 threads 访问连续、对齐的 memory address 时，GPU 可以把这些访问合并成更少的 memory transactions。

## Intuition

>[!note]
> [[GPU]] 不是每个 thread 要一个数，就单独去 HBM 拿一个数。它更喜欢一次搬一整段连续 memory。
>
> 如果一个 warp 里的 32 个 threads 正好访问一段连续地址，这次访问就很高效。

理想情况：

```text
thread 0  -> x[0]
thread 1  -> x[1]
thread 2  -> x[2]
...
thread 31 -> x[31]
````

这些地址连续，可以被合并成少量 memory transactions。

坏情况：

```
thread 0  -> x[0]
thread 1  -> x[1000]
thread 2  -> x[2000]
...
thread 31 -> x[31000]
```

这些地址很分散，GPU 可能需要发起很多 memory transactions。

## Why It Matters

> [!note]  
> Memory coalescing 的核心作用是减少 HBM traffic，让同样的数据访问用更少的 memory transactions 完成。

它和 [[GPU Memory Bound]] 直接相关：

```
访问连续、对齐
→ fewer memory transactions
→ less HBM traffic
→ better effective bandwidth
```

## 和 Bank Conflict 的区别

|Concept|Where|Problem|
|---|---|---|
|[[Memory Coalescing]]|HBM / global memory|warp 访问的地址不连续，不能合并读取|
|[[Bank Conflict]]|shared memory|多个 threads 同时撞到同一个 bank|

> [!important]  
> Memory coalescing 关心的是 **global memory 怎么读得顺**。  
> Bank conflict 关心的是 **shared memory 里面会不会堵车**。

## 和 Tiling 的关系

> [!note]  
> [[Tiling]] 不只是选择 tile size，还要考虑 tile 在 memory 里的 layout 和 alignment。
> 
> 一个 tile 如果读取时地址连续、对齐，就更容易 coalesce；如果 stride 很怪，即使 tile size 看起来合理，也可能访问效率很差。

## Related

- [[GPU]]
- [[GPU Memory Bound]]
- [[Tiling]]
- [[Bank Conflict]]