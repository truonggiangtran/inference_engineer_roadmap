# Week 1 Review: Model Answers

Compare these with your own answers. Wording can differ, but the reasoning should match.

## 1. The path (`B=2, T=4, hidden=8, H=2, D=4`)

```text
token IDs        [2, 4]          integer indices
embeddings       [2, 4, 8]       rows gathered from the [vocab, 8] table
Q / K / V        [2, 4, 8]       three GEMMs: x @ W^T, H*D = 8
split heads      [2, 2, 4, 4]    view [2,4,2,4] then transpose(1,2): [B, H, T, D]
scores           [2, 2, 4, 4]    Q @ K^T / sqrt(4): [B, H, T_q, T_k]
masked + softmax [2, 2, 4, 4]    + causal mask, softmax over T_k (dim=-1)
weights @ V      [2, 2, 4, 4]    [B, H, T_q, D]
merged heads     [2, 4, 8]       transpose back + contiguous + view
```

## 2. Two different multiplications

- **`Q @ K^T`** compares every query with every key. It contracts over the head dimension `D` and produces *scores* shaped `[T_q, T_k]` per head: one number for each query–key pair. Its cost grows with `T²`.
- **`weights @ V`** contracts over the *key/token* dimension `T_k`. Each query's output is a weighted average of value vectors, shaped `[T_q, D]`.

The first decides *where to look*, and the second decides *what to retrieve*.

## 3. Mistakes and the checks that catch them

| Bug | Symptom | Check |
| --- | --- | --- |
| Softmax over the wrong axis (`dim=-2`) | Rows don't sum to 1 | Day 5 row-sum check; Day 6 SDPA comparison (large error) |
| Missing or flipped mask | Future weights are non-zero | Day 5 upper-triangle-zero check; Day 6 causal-independence check |
| Transpose over the wrong axes (`[B,T,H,D]` used as `[B,H,T,D]`) | Heads and tokens are mixed | Day 4 per-head slice check; Day 6 SDPA comparison |
| No `1/√D` scaling | Weights are too peaked | Day 6 SDPA comparison |
| Unstable softmax | `inf`/`NaN` for large scores | Day 5 test with values near 1000 |

## 4. Review questions

1. **Token, word, and character counts differ** because subword tokenizers split text using learned frequent pieces. Common words become one token, rare words, code, and punctuation become several, and a token can include a leading space. Characters are finer than tokens, which are often finer than words.
2. **Token IDs become vectors** through an embedding lookup. Each ID indexes a row of a learned `[vocab, hidden]` matrix. This is a gather, mathematically equal to a one-hot vector times the matrix, but far cheaper.
3. **Queries, keys, and values** are learned linear projections of each token's hidden vector, shaped `[B, H, T, D]` after the heads are split:
   - The query represents what a token is looking for.
   - The key represents what each token offers for matching.
   - The value is the content that gets mixed into the output.
4. **Scaling and softmax:** dividing by `√D` keeps the variance of the dot products near 1, so softmax doesn't saturate. Softmax runs over the **key** dimension (`dim=-1`), giving each query a probability distribution over the positions it may attend to.
5. **Causal-independence test:** perturb only the last position's query, key, and value. The outputs for all earlier positions must stay unchanged (within tolerance), while the last output changes.

## 5. Example five-minute outline

- Text becomes token IDs, and each ID selects a learned vector.
- Every token produces a query, key, and value through three GEMMs, split into heads.
- Scores are scaled query–key dot products. The causal mask removes future positions.
- Softmax over keys turns each row into weights. The weights average the values.
- Heads are merged and projected back to the hidden size.
- Correctness evidence: rows sum to 1, future weights are zero, the result matches SDPA, and the causal-independence test passes.
