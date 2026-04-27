## Day 11 — Speculative Decoding Experiment

**Draft model:** GPT-2 small (117M params)  
**Target model:** GPT-2 large (774M params)  
**Hardware:** NVIDIA T4 16GB  

---

## Results

| Temperature | Acceptance Rate | Tokens/sec | Speedup |
|------------|----------------|------------|---------|
| Baseline   | N/A            | 30.2       | 1.00x   |
| 0.1        | 53.1%          | 25.2       | 0.84x   |
| 0.5        | 81.9%          | 42.3       | 1.40x   |
| 1.0        | 64.0%          | 28.2       | 0.93x   |

![Results](day11_speculative_decoding.png)

---

## Key Findings

**Finding 1 — Temperature 0.5 is the sweet spot**  
81.9% acceptance rate and 1.40x speedup. Too low (0.1) or too high (1.0)
temperature causes both models to disagree differently -- overhead outweighs gains.

**Finding 2 — Overhead kills gains below 70% acceptance**  
At 0.84x and 0.93x, temperatures 0.1 and 1.0 are slower than baseline.
Validates Day 11 threshold: abandon speculative decoding below ~70% acceptance.

**Finding 3 — Training distribution matters more than text type**  
Creative story text achieved 89% acceptance because GPT-2 trained heavily
on narrative web content. Technical AI vocabulary achieved lowest acceptance
-- underrepresented in GPT-2's 2019 training data.

**Finding 4 -- Structured text wins as expected**  
Revenue report: 100% acceptance at temperature 0.5. Every draft token
accepted. Structured, predictable business text is ideal for speculative decoding.

**Finding 5 -- Same-family model pairing is critical**  
GPT-2 small vs large shows moderate acceptance. Llama 3 8B vs 70B
(same training methodology) would show significantly higher acceptance rates.
Draft model must approximate target model's distribution.

---

## PM Insight

The right question before implementing speculative decoding is not
"is this creative or structured text?" It is "does my draft model's
training distribution match my target model's for this use case?"

If the target model was fine-tuned on proprietary data the draft model
never saw -- acceptance rate drops dramatically regardless of text type.

At >70% acceptance: implement speculative decoding.  
At <70% acceptance: try same-family draft model first.  
Still <70%: abandon it, reallocate GPU memory to larger batch sizes.

---

*Souvik Kundu · AI PM Master Curriculum · Day 11*  
*github.com/souvikkai/souvik-ai-pm-portfolio*

