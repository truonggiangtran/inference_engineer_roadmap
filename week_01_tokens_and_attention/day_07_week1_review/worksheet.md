# Week 1 Review Worksheet

Answer from memory first. Mark any answer you had to look up.

## 1. Draw the path

Draw (on paper or below) the path from token IDs to the attention output. Annotate every tensor with its shape, using `B=2, T=4, hidden=8, H=2, D=4`.

```text
token IDs        [ , ]
embeddings       [ , , ]
Q / K / V        [ , , ]
split heads      [ , , , ]
scores           [ , , , ]
masked + softmax [ , , , ]
weights @ V      [ , , , ]
merged heads     [ , , ]
```

## 2. Two different multiplications

Explain how `Q @ K^T` differs from `weights @ V`. What does each produce, and what are its dimensions?

> 

## 3. Mistakes your checks catch

List three bugs, and name the Day 5 or Day 6 check that would catch each one.

| Bug | Symptom | Check that catches it |
| --- | --- | --- |
| | | |
| | | |
| | | |

## 4. Review questions

1. Why do token count, word count, and character count differ?

   > 

2. How do token IDs become learned vectors?

   > 

3. What are the roles and shapes of queries, keys, and values?

   > 

4. Why scale attention scores, and along which dimension is softmax applied?

   > 

5. What test demonstrates that future tokens cannot influence an earlier causal-attention output?

   > 

## 5. Five-minute explanation

Write the bullet points you would use to explain causal attention to a colleague in five minutes.

- 

## 6. Open questions

What is still unclear? Carry it into Week 2.

- 
