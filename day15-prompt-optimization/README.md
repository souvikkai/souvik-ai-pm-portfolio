# Day 15: Prompt Optimization Experiment

## What this is
Systematic benchmark comparing 5 prompt engineering strategies 
on an AI infrastructure recommendation task.
Measures quality, cost, and quality-per-dollar across strategies.

## Strategies tested
1. Zero-Shot
2. Few-Shot (3 examples)
3. Chain of Thought
4. Structured Output (XML)
5. Constitutional AI (self-check layer)

## Key finding
Quality variance across all 5 strategies was only 0.2 points (3.90–4.10/5).
Structured Output delivered the best quality-per-dollar at $0.00585/call —
22% cheaper than Few-Shot with near-identical quality.

At 100K calls/day, choosing Structured Output over the most expensive 
strategy saves ~$XXX/day ($XXX/year).

## PM insight
The hill climb strategy said start at Few-Shot (highest quality).
The data said Structured Output was already the answer.
Frameworks guide the process — data makes the decision.

## How to run
1. Open in Google Colab
2. Add ANTHROPIC_API_KEY to Colab Secrets
3. Run cells 1-7 in order
4. LLM-as-judge auto-scores all outputs
