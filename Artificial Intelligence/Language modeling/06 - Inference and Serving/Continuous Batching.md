#AI #LanguageModeling

## Problem
- Training: get a dense block of tokens (batch size x sequence length)
- [Inference](<Inference.md>): requests arrive and finish at different times, so you have a ragged array
![static batching](<../../attachments/static%20batching.png>)
## Solution: iteration-level scheduling

- Decode step by step
- Add new requests to the batch as they arrive (so don't have to wait until generation completes)

## Problem

- Batching only works when all sequences have the same dimensionality (right?)
- But each request might have a different length

## Solution: selective batching
- Training: when all sequences of the same length, operate on a $B \times S \times H$ tensor
- But we might have different lengths: $[3, H], [9, H], [5, H]$, etc.
- Attention computation: process each sequence separately
- Non-attention computation: concatenate all the sequences together to $[3 + 9 + 5, H]$
