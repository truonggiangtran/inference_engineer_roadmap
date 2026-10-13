# Day 3: Embeddings, Shapes, Indexing, and Devices

**Time:** 60 minutes

## Goal

Understand an embedding lookup and track the batch, sequence, and hidden dimensions.

## Knowledge

**An embedding table is a learned matrix** `W` shaped `[vocab_size, hidden_size]`. Row `i` is the vector the model learned for token ID `i`. An embedding "layer" doesn't multiply or add anything. It selects rows by index:

```text
ids:     [batch, seq]               integers in [0, vocab_size)
W:       [vocab, hidden]
output:  [batch, seq, hidden]       output[b, t] = W[ids[b, t]]
```

**Indexing compared with matrix multiplication.** Selecting row `i` gives the same result as multiplying a one-hot vector by `W`: `one_hot(i) @ W == W[i]`. The two differ in cost:

- **One-hot matmul:** reads the entire `[vocab, hidden]` table and performs `vocab × hidden` multiply-adds per token.
- **Indexing (a gather):** reads only `hidden` values per token.

With Qwen3's vocabulary of about 150,000 entries, indexing is the efficient way to perform the lookup. On the GPU, an embedding lookup is a memory-bound gather, not compute work.

**Read shapes from the outside in.** For `out = emb(ids)` with `ids.shape == [2, 4]` and `hidden = 8`:

| Expression | Shape | Meaning |
| --- | --- | --- |
| `out` | `[2, 4, 8]` | Every token of every sequence |
| `out[0]` | `[4, 8]` | All tokens of sequence 0 |
| `out[0, 2]` | `[8]` | One token's vector |
| `out[:, -1]` | `[2, 8]` | The last token of each sequence |

**Devices must match.** `emb.weight` and `ids` must be on the same device. Move the whole module with `emb.to("cuda")` and the indices with `ids.to("cuda")`. Indices must use an integer dtype; a float index tensor raises an error.

## References

- [PyTorch tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html): indexing and slicing.
- [`torch.nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html): input and output shapes.
- [Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/): the embedding discussion. GPT-2 is a conceptual introduction, not Qwen3's exact architecture.

## Practice

Complete [exercise.py](exercise.py):

1. `make_embedding()`: create a seeded `nn.Embedding(16, 8)`.
2. `lookup()`: look up IDs shaped `[2, 4]`, and confirm that the output is `[2, 4, 8]` and equals `weight[ids]`.
3. `one_hot_matmul()`: compute the same output with `one_hot(ids) @ weight` to show the equivalence, then explain why this version is wasteful.
4. Inspect one token, one sequence, and one batch element. The script prints the shapes.
5. `lookup_on_device()`: repeat the lookup on the GPU if one is available, and compare with the CPU result.

```bash
python week_01_tokens_and_attention/day_03_embeddings_and_shapes/exercise.py
```

## Done when

You can describe how integer IDs select learned vectors and account for every output dimension.
