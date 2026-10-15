# Day 2: Tokenization, Vocabulary, and Python Containers

**Time:** 60 minutes

## Goal

Understand why a language model consumes token IDs rather than words or raw characters.

## Knowledge

**Why tokens?** A model has a fixed-size input layer, the embedding table (Day 3), with one row per vocabulary entry. Each approach to splitting text has tradeoffs:

- **Words:** The vocabulary would be unbounded, because new names, typos, and code identifiers keep appearing.
- **Characters or bytes:** Sequences become very long, and attention cost grows with sequence length.
- **Subword tokens (BPE):** A compromise. Frequent strings become single tokens, while rare strings are split into pieces. Qwen3 uses byte-level BPE, so any input string can be encoded with no unknown tokens.

So **token count ≠ word count**. Code, punctuation, repeated spaces, and non-English text usually need more tokens per word.

**The tokenizer belongs to the model.** Token ID 1234 is just a row number in *this* model's embedding table. If you use a different tokenizer, the same text becomes the wrong rows and the model receives nonsense. Always load the tokenizer and model from the same identifier and revision.

**Tokenizer outputs** are a dict-like `BatchEncoding`:

| Key | Contents |
| --- | --- |
| `input_ids` | Token IDs (list or tensor) |
| `attention_mask` | 1 for real tokens, 0 for padding |

`convert_ids_to_tokens` shows the raw token pieces. Byte-level BPE shows a leading space as `Ġ`. `decode` turns IDs back into text.

**Special tokens and chat templates.** Tokenization turns a string into IDs. A *chat template* first formats a conversation into a string with role markers such as `<|im_start|>user`, and that formatted string is then tokenized. Instruction models expect the template. Qwen3 also supports a "thinking" mode, which this study disables with `enable_thinking=False` so the output format stays consistent.

**Python containers used today:**

- `list`: an ordered sequence, such as token IDs.
- `dict`: a mapping from keys to values, such as the tokenizer output.

## References

- [Hugging Face tokenizer documentation](https://huggingface.co/docs/transformers/main_classes/tokenizer): `__call__`, `decode`, `convert_ids_to_tokens`, and special tokens.
- [Python data structures](https://docs.python.org/3/tutorial/datastructures.html): lists and dictionaries.
- [Qwen3 model documentation](https://huggingface.co/docs/transformers/model_doc/qwen3): tokenizer and chat usage.

## Practice

Complete [exercise.py](exercise.py). It downloads only the tokenizer files for `Qwen/Qwen3-0.6B`, not the model weights.

1. `describe_encoding()`: return the IDs, token pieces, token count, word count, attention mask, and decoded text for one string.
2. `print_encoding()`: print those fields readably.
3. `chat_template_ids()`: format a one-message conversation with the chat template (`enable_thinking=False`), then compare its length with tokenizing the raw text.

```bash
python week_01_tokens_and_attention/day_02_tokenization_and_containers/exercise.py
```

The script tokenizes a sentence, a code fragment, and text with punctuation and repeated spaces. It saves `results/tokenization.json`.

## Done when

You can explain why token count differs from word count and why the tokenizer and vocabulary must match the model.
