#AI #LanguageModeling #Transformer

Learning Rate 就是 [Optimizer](<./Optimizer.md>) 中 gradient decent 的 step size

在训练 [Transformer](<./Transformer.md>) 模型时，能使 Loss 下降最快的 step size 不是固定的，所以我们要做 Learning Rate Schedule。一般是：

起初使用较大的 learning rate，使模型在训练初期进行更快速的更新，然后随着模型训练的进行，逐渐将 learning rate 降低到较小的值。

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
1. the current iteration: $t$
2. the maximum learning rate $\alpha_\max$
3. the minimum (final) learning rate $\alpha_\min$
4. the number of warm-up iterations $T_w$
5. the final iteration of cosine annealing $T_c$.

The learning rate at iteration 𝑡 is defined as:
**(Warm-up)** If $𝑡<T_w$, then $\alpha_t=\frac{t}{T_w}\alpha_\max$
**(Cosine annealing)** If $T_w \leq t \leq T_c$, then $\alpha_t=\alpha_\min+\frac{1}{2}\left(1+\cos\left(\frac{t-Tw}{T_c-T_w}\pi\right)\right)(\alpha_\max-\alpha_\min)$
**(Post-annealing)** If $t>T_c$, then $\alpha_t=\alpha_\min$