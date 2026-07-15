#AI #LanguageModeling #GPU

## What Model Parallelism is..

- It splits up the parameters across GPUs
- But communicate [Activations](<../../Neural%20Networks/Activations.md>) (while [ZeRO](<./ZeRO.md>)3 sends params)

## There are three different types
### [Pipeline Parallelism](<./Pipeline%20Parallelism.md>)
### [Tensor Parallelism](<./Tensor%20Parallelism.md>)
### [Expert parallelism](<./Expert%20parallelism.md>)
