#AI #LanguageModeling #Transformer

Learning Rate 就是 [Optimizer](<./Optimizer.md>) 中 gradient decent 的 step size

在训练 [Transformer](<./Transformer.md>) 模型时，能使 Loss 下降最快的 step size 不是固定的，不同 training stages 适合的 update scale 不同，因此 learning rate 通常随 optimizer step 变化。所以我们要做 Learning Rate Schedule。一般是：

```math
\alpha_t = S(t;\alpha_{\max},\alpha_{\min},T_w,T_c)
```

其中训练前确定：
- $\alpha_{\max}$：peak learning rate；
- $\alpha_{\min}$：final/minimum learning rate；
- $T_w$：warmup steps；
- $T_c$：cosine decay 结束的 optimizer step。

每次 optimizer step，把当前 $t$ 代入已经确定的 schedule，得到这一步的 $\alpha_t$。

# Cosine annealing 余弦退火

```
step size
   ^
max|       ╭────╮
   |      ╱      ╲
   |     ╱        ╲
min|____╱          ╰────────
   +------------------------> t
       warmup   cosine   post
```

The cosine annealing learning rate schedule takes
1. the current iteration: $t$  --- optimizer step
2. the maximum learning rate $\alpha_\max$
3. the minimum (final) learning rate $\alpha_\min$
4. the number of warm-up iterations $T_w$
5. the final iteration of cosine annealing $T_c$.

The learning rate at iteration 𝑡 is defined as:
**(Warm-up)** If $𝑡<T_w$, then $\alpha_t=\frac{t}{T_w}\alpha_\max$
**(Cosine annealing)** If $T_w \leq t \leq T_c$, then $\alpha_t=\alpha_\min+\frac{1}{2}\left(1+\cos\left(\frac{t-T_w}{T_c-T_w}\pi\right)\right)(\alpha_\max-\alpha_\min)$
**(Post-annealing)** If $t>T_c$, then $\alpha_t=\alpha_\min$
