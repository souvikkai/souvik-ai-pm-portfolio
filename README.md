# Souvik Kundu — AI PM Portfolio

Hardware-Native AI PM · Semiconductors × AI Infrastructure  
Senior PM at ONSEMI · UCLA Anderson MBA · MS EE

This repo documents hands-on AI infrastructure experiments, 
silicon teardowns, and PM-level analysis — built as part of 
a 6-week intensive AI PM curriculum.

---

## Portfolio Projects

| Project | Description | Live Demo |
|---------|-------------|-----------|
| P7 — Transformer Attention Visualizer | Scaled dot-product attention computed from scratch. Click any token to inspect its attention distribution. | [Live](https://souvikkai.github.io/souvik-ai-pm-portfolio/p7-attention-visualizer/) |
| Day 7 — SFT Training Experiment | Tiny transformer fine-tuned on 10 AI PM Q&A pairs. Loss reduced 89% over 200 epochs. Same mechanism as ChatGPT SFT at 1/24,000th the scale. | [Notebook](day7-sft-experiment/) |
| Day 5 — Inference Silicon Teardown | PM-level competitive analysis of 7 inference silicon companies — NVIDIA, Groq, Cerebras, SambaNova, d-Matrix, Tenstorrent, Tensordyne. | [Analysis](day5-silicon-teardown/) |
| Day 8 — MMLU Benchmark | Evaluated flan-t5-base across 5 MMLU subjects. Overall 32% vs GPT-4 86%. Math and ethics at random baseline — reveals what scale buys. | [Results](day8-mmlu-benchmark/) |
| Day 9 — Inference Benchmark | Raw PyTorch vs vLLM on phi-2. vLLM delivers 5.5x throughput at 5 concurrent users. Cold start adds 1500ms — never scale to zero. | [Results](day9-inference-benchmark/) |
| Day 11 — Speculative Decoding | GPT-2 small vs large. Temperature 0.5 sweet spot: 81.9% acceptance, 1.40x speedup. Validates 70% acceptance threshold. Training distribution matters more than text type. | [Results](day11-speculative-decoding/) |
| Product 3 — RAG Inference Optimizer | Upload any resume + paste any JD → retrieves relevant experience → benchmarks Claude Sonnet vs Llama 3.1 8B on Groq with Cohere reranking. LLM-as-judge quality scoring. 99% cost reduction at 92% quality. | [Live](https://rag-inference-optimizer.vercel.app/) |
*More projects shipping weekly.*

---

## Background
- Production CNN classifier on Google Vertex AI (99% recall, 100% lot coverage)
- ONSEMI–NVIDIA 800VDC Blackwell power architecture collaboration
- Intel 18A PDK delivery to Big Tech
- MBA UCLA Anderson · MS EE University of Cincinnati

[LinkedIn](https://linkedin.com/in/souvikkundu1) · [GitHub](https://github.com/souvikkai)
