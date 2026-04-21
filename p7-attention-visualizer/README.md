## P7 — Transformer Attention Visualizer

Interactive visualization of scaled dot-product attention computed 
from scratch — no pre-trained model weights, pure NumPy math.

What it shows: How each token distributes attention across the 
full context window. Click any token to inspect its attention distribution.

The math: Q·K^T / √d → softmax → weighted V sum.
Computed across a 10-token coreference sentence.

PM insight: Attention is O(n²) compute and O(n) memory per layer.
KV cache stores K and V vectors to avoid recomputation — which is why 
1M token context windows consume tens of GB of GPU memory per active session.

Live demo: [souvikkai.github.io/souvik-ai-pm-portfolio/p7-attention-visualizer](https://souvikkai.github.io/souvik-ai-pm-portfolio/p7-attention-visualizer)
