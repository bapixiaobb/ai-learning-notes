#AI #LanguageModeling

![gmqa](<../../attachments/gmqa.png>)
👈 是 [Multi-Head Attention](<Multi-Head%20Attention.md>) 有完整的 $H$ 个 [query, key and value heads](<Query%20Key%20Value.md>)
👉 是 Multi-Query Attention 有 $H$ 个 query heads 但是只有 1 个 key 和 value head
👆是 [Grouped-Query Attention](<Grouped-Query%20Attention.md>) 有 $N$ query heads, 但是 $K$ key 和 value heads (interacting with N/K query heads)

- Multi-headed attention (MHA): K=N
- Multi-query attention (MQA): K=1
- Group-query attention (GQA): K is somewhere in between
