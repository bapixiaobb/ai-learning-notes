#GPU

矩阵 Tile 层面的乘加，这是 [GPU](<GPU.md>) 的 [Tensor Core](<Matmul/Tensor%20Core.md>) 提供的，将 scalar [FMA](<../Fundamentals/Fused%20Multiply-Add.md>) 抽象层级提高的层级

```math
D_{m\times n} = A_{m\times k}B_{k\times n} + C_{m\times n}
```
