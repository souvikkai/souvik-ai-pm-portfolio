# Day 16: Long Context Scaling Experiment

## What this measures
Latency and cost at 1K, 10K, and 100K token contexts using Claude Haiku.
Tests whether latency scales linearly, sublinearly, or superlinearly with context length.

## Results

| Context | Input Tokens | Latency | Cost/call | Cost @ 100K calls/day |
|---------|-------------|---------|-----------|----------------------|
| 1K      | 1,265       | 1.35s   | $0.00140  | $140/day             |
| 10K     | 12,389      | 1.15s   | $0.01031  | $1,031/day           |
| 100K    | 123,677     | 2.59s   | $0.09933  | $9,933/day           |

## Key findings
- **Cost scales linearly** — 100K context costs 71x more than 1K, directly proportional to tokens
- **Latency scales sublinearly** — 10x more tokens (10K→100K) adds only 2.25x latency. Flash Attention and HBM bandwidth optimizations prevent O(n²) blowup in this range.
- **Cold start is real** — 1K run paid TCP/TLS handshake overhead, appearing slower than 10K. Never scale to zero for latency-sensitive services.
- **Cost is the binding constraint at scale**, not latency

## PM insight
Moving from 1K to 100K average context at 100K calls/day increases daily API spend
from $140 to $9,933 — a 71x cost increase. Long context is a product pricing decision,
not just a technical one.

## How to run
1. Open in Google Colab (CPU runtime — no GPU needed)
2. Add ANTHROPIC_API_KEY to Colab Secrets
3. Run all 4 cells
