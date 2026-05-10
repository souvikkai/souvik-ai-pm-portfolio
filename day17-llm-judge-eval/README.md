# Day 17: LLM-as-Judge Eval Pipeline

Empirical eval comparing Claude Haiku vs Llama 3.1 8B on a faithfulness-critical RAG task — generating interview talking points from a real resume and job description. Judged by Claude Sonnet 4 across four dimensions.

## Eval Dimensions
- **Faithfulness** — is each talking point grounded in the actual resume?
- **JD Relevance** — does it address a specific JD requirement?
- **Specificity** — concrete numbers, named technologies, measurable outcomes?
- **Interview Utility** — would it genuinely help in a real interview?

## Results

| Model | Faithfulness | JD Relevance | Specificity | Interview Utility | Total | Cost/Run |
|---|---|---|---|---|---|---|
| Claude Haiku | 5/5 | 5/5 | 5/5 | 4/5 | 19/20 | $0.00414 |
| Llama 3.1 8B (Groq) | 2/5 | 4/5 | 4/5 | 1/5 | 11/20 | $0.000108 |

## Key Finding
Llama 3.1 8B is 38x cheaper but fails faithfulness on 60% of outputs — hallucinating resume experience the candidate doesn't have. For faithfulness-critical RAG applications, cost optimization cannot override quality floors. Haiku at $0.004/run is the correct model choice for this use case.

## Stack
- Generator models: Claude Haiku 4.5, Llama 3.1 8B via Groq LPU
- Judge model: Claude Sonnet 4.6
- Eval pattern: LLM-as-judge with binary Pass/Fail per dimension
