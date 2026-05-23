# P2 — Inference Benchmark Suite

## Chapter 1 — Open Source vs Closed API Inference

Structured benchmark comparing:

- Llama 3.1 8B (local inference via Ollama)
- Claude Sonnet (frontier closed API model)

The benchmark evaluates:
- response quality
- reasoning depth
- infrastructure awareness
- latency
- deployment economics
- operational tradeoffs

---

## Benchmark Goals

This benchmark was designed from an AI Infrastructure PM perspective rather than a pure model-evaluation perspective.

Primary questions:
1. Which workloads are “good enough” for local open-source inference?
2. Where do frontier closed models materially outperform?
3. When does self-hosted inference become economically attractive?
4. What infrastructure tradeoffs emerge at scale?

---

## Models Evaluated

| Model | Deployment Type | Notes |
|---|---|---|
| Llama 3.1 8B | Local via Ollama | Open-source self-hosted inference |
| Claude Sonnet | Closed API | Frontier reasoning model |

---

## Prompt Categories

| Category | Count |
|---|---|
| Simple factual | 2 |
| Complex reasoning | 4 |
| AI infrastructure systems | 4 |

---

## Evaluation Rubric

Responses were scored from 1–5 across:
- Accuracy
- Depth
- Conciseness

Detailed rubric:
- `docs/scoring_rubric.md`

---

## Key Early Findings

### 1. Frontier API models significantly outperform small local OSS models on infrastructure reasoning

Llama 3.1 8B performed well on:
- simple factual explanations
- lightweight PM reasoning
- concise conceptual summaries

However, performance degraded materially on:
- production inference architecture
- memory hierarchy reasoning
- RAG observability
- deployment economics
- semiconductor systems tradeoffs

Claude consistently demonstrated:
- stronger systems-level reasoning
- better operational realism
- deeper production infrastructure awareness
- more nuanced PM tradeoff analysis

The gap widened substantially on prompts involving:
- KV-cache optimization
- hybrid retrieval pipelines
- inference economics
- GPU architecture tradeoffs

---

### 2. Open-source local inference remains viable for lightweight internal workloads

Despite weaker reasoning depth, Llama 3.1 8B remained usable for:
- FAQ-style retrieval
- lightweight summarization
- basic copilots
- internal productivity workflows
- low-risk enterprise automation

The model performed best when:
- prompts were narrow and factual
- domain ambiguity was low
- infrastructure reasoning depth was not required

This supports the idea that smaller OSS models can meaningfully reduce inference cost for constrained workloads.

---

### 3. The largest quality gap was not factual accuracy — it was systems nuance

In many prompts, Llama produced technically correct answers.

The primary difference was:
- operational realism
- systems thinking
- infrastructure tradeoff awareness
- production deployment nuance

This suggests that frontier models may derive substantial advantage from:
- broader systems-training distribution
- higher-quality reasoning alignment
- richer infrastructure-related latent knowledge

---

### 4. Inference infrastructure is increasingly memory-bandwidth bound rather than FLOPS bound

Several benchmark prompts reinforced a recurring infrastructure theme:
modern LLM inference bottlenecks are dominated by:
- HBM bandwidth
- KV-cache growth
- memory movement
- cache locality

rather than raw compute throughput alone.

This has major implications for:
- accelerator design
- inference serving architectures
- batching strategies
- quantization approaches
- next-generation AI hardware startups

## Aggregate Benchmark Scores

| Model | Avg Accuracy | Avg Depth | Avg Conciseness |
|---|---|---|---|
| Llama 3.1 8B | 3.6 | 3.3 | 4.0 |
| Claude Sonnet | 5.0 | 5.0 | 4.9 |

---

## Category-Level Observations

| Category | Llama 3.1 8B | Claude Sonnet |
|---|---|---|
| Simple factual | Strong | Excellent |
| Complex reasoning | Moderate | Excellent |
| AI infrastructure systems | Weak–Moderate | Excellent |

---

## Latency Observations

| Model | Typical Latency |
|---|---|
| Llama 3.1 8B (local RTX laptop) | ~7–52s depending on prompt complexity |
| Claude Sonnet (API/web) | ~3–12s estimated |

Latency variance on local inference increased significantly for:
- long-context prompts
- systems reasoning prompts
- architecture-heavy explanations

This reflects the compute and memory limitations of small local inference environments compared to frontier hosted infrastructure.

---

## Cost Analysis

### Assumptions

#### Local Inference — Llama 3.1 8B
- RTX laptop local inference via Ollama
- Approximate system power draw during inference: 120W
- Electricity cost: $0.12 / kWh
- Hardware amortization assumption:
  - $1800 laptop
  - amortized over 3 years
  - ~3 inference hours/day average utilization

#### Frontier API Inference — Claude Sonnet
Assumptions:
- Average prompt: 700 input tokens
- Average response: 500 output tokens
- 1000 benchmark-style queries

Claude pricing varies over time and region; calculations below are directional PM estimates rather than exact billing forecasts.

---

### Local Inference Cost Estimate

#### Electricity Cost

120W × 1 hour = 0.12 kWh

0.12 kWh × $0.12 = ~$0.014/hour

Assuming:
- ~35 seconds average/query
- 1000 queries
- ~9.7 GPU-hours total runtime

Estimated electricity cost:
~$0.14

---

#### Hardware Amortization

$1800 laptop / 3 years ≈ $600/year

Assuming:
- 3 hours/day inference utilization
- ~1095 inference-hours/year

Approximate hardware cost:
~$0.55/hour

For ~9.7 runtime hours:
~$5.34

---

### Total Local Inference Cost

| Component | Estimated Cost |
|---|---|
| Electricity | ~$0.14 |
| Hardware amortization | ~$5.34 |
| Total | ~$5.48 per 1000 queries |

---

### Frontier API Cost Estimate

Assuming:
- 700 input tokens/query
- 500 output tokens/query
- 1000 queries

Approximate token volume:
- 700K input tokens
- 500K output tokens

Estimated frontier API cost:
~$8–25 per 1000 queries depending on model pricing tier.

---

### Economic Insight

At small scale, API inference is operationally superior:
- no infrastructure management
- elastic scaling
- frontier-quality reasoning

However, at sustained high-volume workloads:
- fixed infrastructure costs amortize
- batching efficiency improves
- quantization reduces serving cost
- OSS inference becomes economically attractive

The crossover point depends heavily on:
- request volume
- average token length
- concurrency
- latency SLA
- model size
- hardware utilization efficiency

---

## PM Implications

### Recommended Deployment Strategy by Company Stage

| Company Stage | Recommended Strategy | Why |
|---|---|---|
| Seed | Frontier API-first | Maximize iteration speed, minimize infra burden |
| Series A | Hybrid architecture | Move predictable/high-volume workloads to OSS inference |
| Series B+ | Dedicated inference infrastructure | Optimize unit economics, latency, and workload specialization |

---

### Seed Stage — API First

For early-stage startups, frontier APIs are usually the correct decision because:
- product iteration speed matters more than infra optimization
- engineering teams are small
- workloads are still poorly understood
- traffic patterns are unstable
- reliability and reasoning quality dominate infrastructure cost

The primary risk at this stage is over-investing in infrastructure before product-market fit is established.

---

### Series A — Hybrid Transition

As workloads stabilize, teams gain visibility into:
- token distribution
- concurrency patterns
- latency requirements
- workload segmentation

At this stage, the optimal architecture is often hybrid:
- frontier APIs for high-complexity reasoning
- OSS/self-hosted inference for:
  - embeddings
  - summarization
  - classification
  - retrieval
  - lightweight copilots

This creates meaningful cost leverage without sacrificing frontier capability where it matters most.

---

### Series B+ — Infrastructure Optimization Becomes Strategic

At scale:
- inference becomes a major COGS driver
- GPU utilization becomes critical
- batching efficiency matters materially
- serving architecture impacts gross margin

At this stage, companies begin investing in:
- quantization
- speculative decoding
- custom serving stacks
- multi-model routing
- KV-cache optimization
- dedicated inference clusters

Infrastructure becomes a competitive advantage rather than a backend implementation detail.

---

### Key PM Insight

The correct deployment strategy is not:
- “open source vs closed source”

It is:
- workload-model alignment.

Different workloads require different tradeoffs across:
- latency
- reasoning quality
- cost
- privacy
- reliability
- operational complexity

The strongest AI infrastructure organizations increasingly operate heterogeneous inference stacks rather than relying on a single model provider.

---

## Hardware-Native AI PM Positioning

This benchmark was designed from the perspective of a hardware-native AI Product Manager rather than a pure ML researcher.

The goal was not simply to compare model outputs, but to evaluate the infrastructure and product tradeoffs between:
- frontier hosted inference
- open-source self-hosted inference

Key themes explored:
- memory-bandwidth bottlenecks
- inference economics
- GPU utilization
- KV-cache scaling
- retrieval architecture
- deployment lifecycle strategy
- accelerator architecture tradeoffs

This work directly connects semiconductor systems thinking with modern AI infrastructure product management.

The benchmark aligns closely with the types of infrastructure decisions faced by companies such as:
- Groq
- CoreWeave
- SambaNova
- NVIDIA
- hyperscale AI platform providers

particularly around:
- serving economics
- accelerator utilization
- workload-model alignment
- infrastructure scalability
- inference optimization

## Methodology & Limitations

### Methodology

- Local inference executed using Ollama on an RTX-enabled laptop
- Frontier model responses generated using Claude Sonnet
- All models evaluated using identical prompts
- Responses scored manually across:
  - Accuracy
  - Depth
  - Conciseness

Evaluation combined:
- human PM judgment
- systems-level infrastructure reasoning assessment
- qualitative production realism analysis

---

### Important Limitation

This benchmark intentionally does NOT compare parameter-matched models.

The objective was to compare:
- deployment archetypes
- operational tradeoffs
- infrastructure implications

rather than raw parameter-scale parity.

Accordingly:
- Llama 3.1 8B represents lightweight OSS local inference
- Claude Sonnet represents frontier hosted inference

This reflects a realistic product decision many AI startups face in practice.

---

### Additional Notes

LLM-as-a-judge approaches may exhibit bias toward frontier model writing styles and reasoning patterns.

To mitigate this:
- qualitative human review was also performed
- scoring emphasized operational realism and systems thinking rather than prose sophistication alone
