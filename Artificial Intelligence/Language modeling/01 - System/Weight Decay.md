#AI #LanguageModeling #Optimization

# Weight Decay

[Small-norm Preference](<Small-norm%20Preference.md>) 解释为什么偏好较小的 parameter norm；weight decay 描述怎样在 optimizer step 中实现这种偏好。

数学上，small-norm preference 可以写成：

```math
\min_\theta
\left[
\mathcal L_{\text{data}}(\theta)
+
\frac{\lambda}{2}\|\theta\|_2^2
\right]
```

Weight decay 在每一步直接缩小当前 parameters：

```math
\theta_t
\leftarrow
(1-\alpha_t\lambda)\theta_t
```

- $\alpha_t$：learning rate；
- $\lambda$：weight decay coefficient。

如果没有 gradient update，反复执行 weight decay 会让 parameters 逐渐趋近于 0。实际训练中，gradient update 与 weight decay 同时作用，因此有助于降低 loss 的 parameters 不会简单地全部变成 0。

AdamW 和 Muon 都可以把 weight decay 与各自的 gradient update 分开执行。
