#AI #LanguageModeling 


>[!note] 这是 [[GPU]] 在数据访问的时候会出现的问题
Bank conflict 发生在 shared memory。shared memory 被分成多个 banks，如果同一个 warp 里的多个 threads 同时访问同一个 bank，就会排队，导致 shared memory 访问变慢。

用一句话总结就是：Bank conflict 是 shared memory 里面的访问堵车

GPU 硬件中：

```
HBM 很慢
shared memory 很快
```

但是

>shared memory 也不是一整块魔法快内存。它内部又被切成很多个 bank。这个 bank 就是访问内存的窗口

可以把 bank 理解成银行柜台，shared memory 就是银行金库

```
shared memory = 32 个小窗口 / 小柜台
bank 0
bank 1
bank 2
...
bank 31
```

一个 warp 有 32 个 threads。理想情况是：

```
thread 0 访问 bank 0
thread 1 访问 bank 1
thread 2 访问 bank 2
...
thread 31 访问 bank 31
```

这样 32 个 threads 可以同时拿到数据，很快。

坏情况是：

```
thread 0 访问 bank 0
thread 1 也访问 bank 0
thread 2 也访问 bank 0
...
thread 31 也访问 bank 0
```

那 bank 0 一个时刻处理不过来，只能排队。

这就叫 **bank conflict**。这个时候同一个 warp 里的多个 threads 同时访问 shared memory 的同一个 bank，导致访问被串行化。