# Day 22 — QLoRA Fine-Tuning Lab

Hands-on exploration of QLoRA fine-tuning using a quantized instruct model on a constrained GPU environment (Tesla T4).

Goal:  
Understand the systems + infrastructure tradeoffs behind parameter-efficient fine-tuning rather than simply training a chatbot.

---

## What I Built

- Loaded a 4-bit quantized Qwen2.5-0.5B-Instruct model
- Applied LoRA adapters using PEFT
- Fine-tuned on a small AI infrastructure domain dataset
- Trained successfully on a Tesla T4 GPU
- Saved lightweight deployable LoRA adapters
- Debugged mixed precision + dtype compatibility issues

---

## Key Concepts Explored

- LoRA vs QLoRA
- NF4 quantization
- fp16 vs bf16
- Transformer KV cache
- Parameter-efficient fine-tuning (PEFT)
- Gradient accumulation
- Tokenizer/chat-template alignment
- LoRA vs RAG tradeoffs

---

## Stack

| Component | Tool |
|---|---|
| Base Model | Qwen2.5-0.5B-Instruct |
| Fine-Tuning | QLoRA |
| Quantization | 4-bit NF4 |
| Framework | Hugging Face Transformers |
| PEFT | LoRA adapters |
| Trainer | TRL SFTTrainer |
| GPU | Tesla T4 |
| Environment | Google Colab |

---

## Results

### Parameter Efficiency

```text
Trainable params: 540,672
Total params: 494,573,440
Trainable %: 0.1093%
```

Only ~0.11% of model parameters were updated during training.

---

### GPU Memory Usage

```text
GPU: Tesla T4
Max allocated memory: ~1.08 GB
Max reserved memory: ~1.53 GB
```

Despite training a ~0.5B parameter model, QLoRA kept memory usage extremely low through:
- 4-bit quantization
- frozen base weights
- lightweight LoRA adapters

---

## Important Learnings

### 1. Lower Loss ≠ Better Outputs

Training loss decreased successfully:

| Step | Loss |
|---|---|
| 1 | 5.92 |
| 5 | 5.37 |

However, generation quality was still weak because the dataset was extremely small.

The model learned:
- domain vocabulary
- formatting patterns

But struggled with:
- deep semantic understanding
- accurate KV cache reasoning

A useful reminder that:
> small datasets often teach style before knowledge.

---

### 2. Chat Templates Matter

Initial training used a generic:

```text
### Question / ### Answer
```

format.

This was replaced with Qwen’s native chat template using:

```python
tokenizer.apply_chat_template(...)
```

Without proper template alignment, instruct models can produce unstable or degraded outputs.

---

### 3. Hardware Compatibility Matters

Training initially failed with:

```text
NotImplementedError:
... not implemented for 'BFloat16'
```

Root cause:
- LoRA trainable parameters were initialized in `bfloat16`
- Tesla T4 did not support this AMP configuration safely

Fix:
- manually converted trainable LoRA params to `float32`

This was a good reminder that:
> infrastructure constraints directly affect ML training behavior.

---

## LoRA vs RAG Takeaway

This project reinforced an important production insight.

### LoRA is useful for:
- behavior steering
- formatting
- lightweight specialization
- tone adaptation

### RAG is often better for:
- factual retrieval
- evolving knowledge
- enterprise documents
- technical correctness

Fine-tuning is not always the right solution.

---

## Saved Artifacts

```text
adapter_model.safetensors ≈ 2.1 MB
compressed adapter package ≈ 4 MB
```

The trained adapter is tiny compared to the full base model, making:
- deployment cheaper
- experimentation faster
- multi-tenant customization practical

---

## Key PM / Infrastructure Takeaways

- Quantization fundamentally changes GPU economics
- PEFT enables cheap experimentation loops
- Better data matters more than larger models
- Tokenizer/template alignment is critical
- Training loss alone is not a reliable product metric
- Notebook reproducibility can become a real engineering problem

---

## Future Improvements

- Larger curated dataset
- Train/eval split
- Proper evaluation harness
- Latency benchmarking
- LoRA vs RAG comparison
- Adapter merging experiments
- Hallucination analysis
