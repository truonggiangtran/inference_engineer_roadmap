# Final Review and Next-Step Decision

## 1. Ten-minute architecture walkthrough

Deliver it out loud, without notes, then write down what you missed or hesitated on.

- Missed or uncertain:

## 2. Five-minute performance explanation

Cover the cache comparison, the numerical checks, one bottleneck hypothesis, and the evidence that would change your conclusion.

- Missed or uncertain:

## 3. Week 4 review questions

1. What was held constant, and what changed in your cache comparison?

   > 

2. How did you establish numerical correctness before timing?

   > 

3. What exactly do your latency and throughput numbers measure?

   > 

4. Which observation supports each performance explanation?

   > 

5. What limitation prevents generalizing the result to all models, GPUs, or serving workloads?

   > 

6. What is the smallest next experiment that could test your strongest remaining hypothesis?

   > 

## 4. Self-assessment

Rate your confidence for each area as 1 (weak) to 3 (strong), then note the gap.

| Area | Confidence | Gap |
| --- | --- | --- |
| Tokens, embeddings, attention math | | |
| Real decoder block (RoPE, GQA, RMSNorm, MLP) | | |
| Prefill vs cached decode | | |
| GPU timing and profiling | | |
| Compute vs bandwidth vs overhead reasoning | | |
| Memory accounting | | |
| Experimental design and reporting | | |

## 5. Next project decision

- **Chosen project** (fused RMSNorm/residual, deeper attention profiling, or a runtime experiment):
- **The gap in my study that motivates it:**
- **Hypothesis it tests:**
- **Correctness check:**
- **Success metric and protocol changes:**
- **Explicitly out of scope** (for example, ROCm claims without AMD hardware):
