## Day 8 — MMLU Benchmark Evaluation

Evaluated google/flan-t5-base (250M parameters) across 5 MMLU 
subjects — 100 questions total.

**Overall score: 32%** vs GPT-4 baseline of ~86%.

| Subject | Score |
|---------|-------|
| Astronomy | 40% |
| High School US History | 35% |
| High School Computer Science | 35% |
| Elementary Mathematics | 25% |
| Moral Scenarios | 25% |

**Key finding:** Elementary mathematics and moral scenarios scored 
at random baseline (25%) — the model is essentially guessing, since it has
no mathematical reasoning and moral scenarios need nuanced understanding, 
which needs RLHF training.
Astronomy performed best at 40% — factual recall tasks favor 
smaller models more than reasoning tasks.

**PM insight:** Aggregate benchmark scores hide per-domain weakness. 
A product requiring mathematical reasoning would fail completely 
with this model despite its above-random overall score. Always 
break benchmarks down by subject before making model selection 
decisions. Build evals for your specific use case — not just 
MMLU overall.

**The scale gap:** 54% separates flan-t5-base from GPT-4. 
That gap is explained by 7,200x more parameters and orders 
of magnitude more training data — not a fundamentally 
different algorithm.
