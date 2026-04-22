# AI Inference Silicon Teardown
## Souvik Kundu · AI PM Master Curriculum · Day 5
### A PM-Level Competitive Analysis of 7 Inference Silicon Companies

---

## The Landscape

Every company below is betting against NVIDIA's dominance in AI inference.
NVIDIA has ~80% market share, a $30-35K H100, and the CUDA ecosystem moat.
Each challenger is making a different architectural bet on where NVIDIA is vulnerable.

The core insight: **NVIDIA optimizes for flexibility. Every challenger optimizes for something specific.**
Flexibility costs efficiency. Specificity wins on cost-per-token at scale.

---

## The Memory Bandwidth Problem

Before the teardown — one concept that explains why all of these companies exist.

Modern LLMs are **memory-bound**, not compute-bound. The GPU spends most of its time
waiting for model weights to move from HBM memory to compute cores — not actually computing.

```
Tokens/sec = Memory Bandwidth (GB/s) ÷ Model Size (GB)

H100 example: 3,350 GB/s ÷ 140 GB (70B in FP16) = ~24 tokens/sec at batch 1
```

Every company below is attacking this bottleneck from a different angle.

---

## Company 1 — NVIDIA

**The incumbent. The moat. The benchmark everyone else is measured against.**

### The Bet
CUDA ecosystem lock-in is deeper than hardware. The switching cost is not the chip price —
it is millions of lines of CUDA-dependent code that every ML team has written over 15 years.

### Architecture
- H100: 80GB HBM3, 3,350 GB/s bandwidth, ~700 TFLOPS BF16
- B200 (Blackwell): 192GB HBM3e, 8,000 GB/s bandwidth, ~2.25 PFLOPS FP8
- GB200 NVL72: 72 B200 GPUs + 36 Grace CPUs, 576 TB/s total bandwidth, rack-scale system
- NVLink for GPU-to-GPU communication at 1.8 TB/s per GPU

### The Value Chain
```
ML Engineer writes PyTorch
→ PyTorch calls CUDA
→ CUDA optimizes execution on NVIDIA GPU
→ NVIDIA captures the value
```

Breaking NVIDIA means breaking this chain. No one has done it at scale.

### The Real Moat
Not the chip. Not the price. **JAX is NVIDIA's long-term risk** — if JAX displaces PyTorch
as the dominant ML framework, the CUDA dependency breaks. Until then, NVIDIA is safe.

### PM Implication
Custom silicon wins for captive workloads at hyperscaler scale — Google trains Gemini on TPUs,
Anthropic trains Claude on Trainium. The broad market stays on NVIDIA because migration cost
exceeds savings for most companies.

---

## Company 2 — Groq

**The latency specialist. Deterministic execution. No surprises.**

### The Bet
NVIDIA GPUs are designed for parallel batch processing. For single-query, real-time inference,
most of that parallelism is wasted. Groq bets that a chip designed specifically for
deterministic, low-latency execution wins for real-time applications.

### Architecture
- **LPU (Language Processing Unit)** — not a GPU
- **230MB SRAM per chip** — all model weights on-chip, no HBM
- **Pre-scheduled compiler** — execution path computed at compile time, not runtime
- No dynamic scheduling overhead — the chip never has to decide what to do next
- Result: deterministic latency — same response time every single query

### The Key Insight
NVIDIA's GPU scheduler adds latency variance because it dynamically decides how to allocate
compute. Groq eliminates the scheduler entirely. Every operation is pre-planned.

### Numbers
- ~500 tokens/sec on Llama 3 70B — significantly faster than H100 for single queries
- Latency is predictable to milliseconds — critical for real-time applications

### The Constraint
230MB SRAM is fast but limited. Large models (70B+) require multiple chips.
Memory ceiling is the PM challenge — you cannot run a 405B model on a small Groq cluster easily.

### PM Implication
Groq wins for: chatbots, real-time voice AI, low-latency APIs, any use case where
P99 latency matters more than throughput. Loses for: large batch training, massive models.

---

## Company 3 — Cerebras

**The wafer-scale bet. Eliminate the memory bottleneck entirely.**

### The Bet
The fundamental problem is moving data between HBM and compute cores.
Cerebras eliminates HBM entirely — put all the memory on the chip itself.

### Architecture
- **WSE-3 (Wafer Scale Engine)** — a single chip the size of an entire silicon wafer
- **900,000 cores** on one chip
- **44GB on-chip SRAM** — no HBM, no memory bandwidth bottleneck
- **21 PB/s memory bandwidth** — orders of magnitude faster than HBM

### Why Wafer Scale
Normal chips are cut from wafers — you get hundreds of small chips per wafer.
Cerebras uses the entire wafer as one chip. Eliminates the packaging and interconnect
overhead between chips. All memory and compute on one substrate.

### The Constraint
44GB is fast but fixed. GPT-4 class models at FP16 need 1TB+.
You need many WSE-3 chips for large models — and connecting wafer-scale chips
introduces the interconnect problem Cerebras was designed to avoid.

**The memory ceiling is the central PM challenge for Cerebras.**
Great for models that fit. Constrained for frontier-scale models.

### PM Implication
Cerebras wins for: mid-size models (7B-34B) where the entire model fits on-chip,
research workloads, customers who need extreme throughput on known model sizes.

---

## Company 4 — SambaNova

**The reconfigurable bet. Hardware that reshapes itself to the model.**

### The Bet
Fixed hardware architectures waste resources on workloads they weren't optimized for.
SambaNova bets that reconfigurable hardware — chips that restructure their dataflow
to match each specific model — wins for enterprise deployments with fixed workloads.

### Architecture
- **RDU (Reconfigurable Dataflow Unit)**
- Hardware reconfigures its internal connections to match the model's computation graph
- Once configured for a specific model, extremely efficient for that model
- Different from GPU: GPU maps computation to fixed hardware, RDU maps hardware to computation

### The Key Insight
Enterprise customers typically run the same model repeatedly — same architecture,
same weights, same query patterns. Reconfigurable hardware can be optimized once
and then run that workload with maximum efficiency indefinitely.

### The Constraint
Reconfiguration takes time. Not ideal for dynamic workloads or rapid model switching.
Best suited for predictable, high-volume enterprise deployments.

### PM Implication
SambaNova wins for: enterprise customers with fixed deployment patterns,
financial services, healthcare, any vertical where the same model runs millions of times.
Loses for: research environments, rapid model iteration, diverse workload mix.

---

## Company 5 — d-Matrix

**The in-memory compute bet. Eliminate the data movement problem at source.**

### The Bet
The memory bandwidth problem exists because compute and memory are separate.
Data moves from memory to compute — that movement is the bottleneck.
d-Matrix bets that performing computation inside the memory itself eliminates the bottleneck.

### Architecture
- **In-memory compute** — arithmetic operations happen where data is stored
- No data movement between memory and compute cores
- Eliminates the roofline model constraint entirely for certain operations
- Matrix multiply — the dominant LLM operation — happens inside the memory array

### Why This Matters
In standard chips: weights move from HBM → cache → compute cores → result moves back.
In d-Matrix: weights never move. Computation happens in place.

### Current Status
Pre-revenue, deep tech. Still in development/early customer phase.
The architecture is theoretically compelling but unproven at production scale.

### PM Implication
Highest potential architectural differentiation of any company on this list.
Also highest execution risk — in-memory compute at scale is genuinely hard.
PM opportunity: whoever figures out the software stack for in-memory compute
wins a significant position in the next generation of inference silicon.

---

## Company 6 — Tenstorrent

**The open architecture bet. RISC-V cores, heterogeneous compute, Jim Keller.**

### The Bet
AI compute should be open and programmable — not locked into proprietary ecosystems.
Tenstorrent bets that RISC-V based, heterogeneous compute cores win because
they can be programmed for diverse workloads including emerging agentic AI patterns.

### Architecture
- **Tensix cores** — custom RISC-V based processor cores
- Heterogeneous compute — mix of different core types on one chip
- Each Tensix core has local memory + compute + a RISC-V processor for control flow
- More flexible than fixed-function accelerators
- Designed for agentic AI workloads with unpredictable control flow

### The Jim Keller Factor
Jim Keller designed the AMD K8 (saved AMD), Apple A4/A5, Tesla FSD chip, and Intel's
roadmap recovery. His presence at Tenstorrent is a significant credibility signal.

### Why Agentic AI Matters for Architecture
Standard LLM inference: predictable token generation, regular memory access patterns.
Agentic AI: tool calls, branching logic, variable context length, unpredictable execution.
Tenstorrent's RISC-V cores handle control flow better than pure matrix accelerators.

### PM Implication
Tenstorrent wins if agentic AI becomes the dominant inference pattern.
The RISC-V open architecture is also a strategic bet against CUDA lock-in —
customers who want programmability without NVIDIA dependency.

---

## Company 7 — Tensordyne (Recogni)

**The logarithmic math bet. Change the arithmetic, change the power equation.**

### The Bet
All AI chips do the same math — floating point multiply-accumulate operations.
Tensordyne bets that changing the number representation from linear to logarithmic
transforms the power efficiency of every AI operation.

### The Core Innovation
In standard arithmetic: a × b requires a multiplier circuit (expensive, power-hungry)
In logarithmic arithmetic: log(a × b) = log(a) + log(b) — multiplication becomes addition

**Multiplication → Addition. Addition is dramatically cheaper in silicon.**

The challenge: addition in log space is not simple. How you handle log(a + b) is
Tensordyne's secret sauce — their proprietary approximation method.

### Numbers
- 8x power efficiency vs GB200 NVL72 on equivalent workloads
- 22x energy reduction per operation vs standard floating point
- Architecture agnostic — works across CNN, transformer, and other model types
- Inference focused — datacenter deployment, not training

### History
- Founded 7 years ago as Recogni — originally for autonomous vehicle inference
- Pivoted to datacenter inference as the larger opportunity
- Offices: Munich (AI/algorithm team) + San Jose (chip design team)
- Partnership with Juniper Networks for networking stack

### The Constraint
Logarithmic approximations introduce numerical errors. The question is whether
those errors affect model accuracy at production quality thresholds.
This is the central PM and engineering challenge — accuracy vs efficiency tradeoff.

### ONSEMI Connection
ONSEMI and NVIDIA are collaborating on 800VDC power architecture for next-gen
Blackwell GPU racks. Tensordyne's 8x power efficiency claim, if validated,
would fundamentally change datacenter power budgeting — directly relevant to
the infrastructure work happening at the silicon level.

### PM Implication
If the accuracy-efficiency tradeoff is solved, Tensordyne's architecture is
a generational shift in inference economics. The model conversion toolchain —
making it easy for customers to convert and validate their models — is the
critical PM problem. Every week of friction in model conversion is a week
closer to the customer choosing to wait for Blackwell instead.

---

## The Competitive Map

| Company | Architectural Bet | Wins When | Loses When |
|---------|------------------|-----------|------------|
| NVIDIA | CUDA ecosystem moat | Everyone needs flexibility | JAX displaces PyTorch |
| Groq | Deterministic, zero-variance latency | Real-time inference matters | Large batch throughput needed |
| Cerebras | Eliminate HBM with wafer-scale SRAM | Model fits on-chip | Models exceed 44GB |
| SambaNova | Reconfigurable hardware per workload | Fixed enterprise deployments | Dynamic workload mix |
| d-Matrix | In-memory compute eliminates data movement | Architecture proven at scale | Execution risk materializes |
| Tenstorrent | RISC-V flexibility for agentic AI | Agents become dominant pattern | Standard LLM inference stays dominant |
| Tensordyne | Log arithmetic cuts power 8x | Accuracy tradeoff solved | Numerical errors affect quality |

---

## The PM Framework — How to Evaluate Any New Inference Silicon Company

Four questions every AI infra PM should ask:

**1. What bottleneck are they attacking?**
Memory bandwidth? Compute? Power? Latency variance? Programmability?

**2. What workload do they win on?**
No chip wins everything. What specific use case is this architecture optimal for?

**3. What is the software ecosystem story?**
Hardware without software is useless. How do customers get their models running?
Model conversion toolchain quality is often the deciding factor.

**4. What is the migration cost from NVIDIA?**
If the answer is "rewrite your entire stack" — most customers won't do it.
If the answer is "run our converter and validate in a day" — adoption accelerates.

---

## Key Takeaways for Interviews

1. **NVIDIA's moat is CUDA, not the chip.** The chip can be beaten on specs.
   The software ecosystem cannot be easily replicated.

2. **Every challenger attacks a specific bottleneck.** Understanding which bottleneck
   tells you which customers they can win.

3. **The software stack is the PM problem.** Not the chip architecture.
   Model conversion, validation tooling, developer experience — this is where
   hardware-native AI PMs create value.

4. **Power efficiency is becoming the constraint.** Datacenter power budgets are hitting
   physical limits. Tensordyne's 8x efficiency claim, if validated, is not incremental —
   it is architectural.

5. **Agentic AI changes the inference requirements.** Unpredictable control flow,
   variable context length, tool use — these patterns favor flexible architectures
   (Tenstorrent) over fixed-function accelerators.

---

*Souvik Kundu · AI PM Master Curriculum · souvik-ai-pm-portfolio*
*github.com/souvikkai · linkedin.com/in/souvikkundu1*
