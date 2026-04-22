## Day 7 — SFT Training Experiment

Tiny transformer fine-tuned on 10 AI PM Q&A pairs from the 
AI PM Master Curriculum. Loss reduced 89% over 200 epochs.

Same SFT mechanism OpenAI used for ChatGPT — at 1/24,000th the scale.

**Key observation:** Loss drops from 3.68 (random weights) to ~0.41 
(learned patterns) — gradient descent working in real time.

![Loss Curve](day7_loss_curve.png)
