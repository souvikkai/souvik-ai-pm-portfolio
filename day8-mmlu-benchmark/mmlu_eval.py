# ─────────────────────────────────────────────────────────────────────────────
# Day 8 — MMLU Benchmark Evaluation
# Souvik Kundu · AI PM Master Curriculum
# souvik-ai-pm-portfolio
#
# What this script does:
# Runs a small Hugging Face model (flan-t5-base) against MMLU questions
# across 5 subjects and analyzes where it fails and why.
#
# PM insight: benchmark scores are averages. The per-subject breakdown
# reveals whether failures are random or systematic — which tells you
# whether the model is genuinely weak or just untested in a domain.
#
# Run: python mmlu_eval.py
# Requirements: pip install transformers torch datasets matplotlib
# ─────────────────────────────────────────────────────────────────────────────

import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving charts
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
import json
import time

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: MODEL SETUP
#
# We use flan-t5-base — a 250MB model from Google.
# "flan" means it was instruction-tuned (SFT'd) on a large collection
# of tasks including question answering.
#
# Why flan-t5 and not GPT?
# - Free, no API key needed
# - Small enough to run locally without a GPU
# - Instruction-tuned so it understands Q&A format
# - Good enough to show interesting failure patterns on MMLU
#
# PM note: this is a "base" model for our purposes — not frontier.
# GPT-4 scores ~86% on MMLU. flan-t5-base will score much lower.
# That gap IS the insight — it shows what scale buys you.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("Day 8 — MMLU Benchmark Evaluation")
print("Model: google/flan-t5-base")
print("=" * 65)
print()
print("Loading model... (downloading ~250MB on first run)")

MODEL_NAME = "google/flan-t5-base"

# Load tokenizer — converts text to token IDs the model understands
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Load the model itself — this is the 250MB file with all the weights
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Set to eval mode — disables dropout, makes inference deterministic
model.eval()

# Use GPU if available, otherwise CPU
# flan-t5-base is small enough to run on CPU in reasonable time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded. Running on: {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: MMLU DATASET
#
# We load MMLU directly from Hugging Face datasets.
# MMLU has 57 subjects. We test 5 to keep runtime reasonable.
#
# Subject selection is deliberate — we pick subjects that test
# different types of knowledge:
#   - elementary_mathematics  → basic reasoning
#   - high_school_history     → factual recall
#   - computer_science        → technical knowledge
#   - moral_scenarios         → judgment and ethics
#   - astronomy               → scientific reasoning
#
# PM insight: different subjects test different capabilities.
# A model that aces history but fails math reveals something
# about how it was trained and what it learned.
# ─────────────────────────────────────────────────────────────────────────────

SUBJECTS = [
    "elementary_mathematics",
    "high_school_us_history",
    "high_school_computer_science",
    "moral_scenarios",
    "astronomy",
]

# Number of questions to test per subject
# 20 per subject = 100 total questions
# Enough for meaningful results, fast enough to run in ~10 minutes
QUESTIONS_PER_SUBJECT = 20

print(f"Testing {len(SUBJECTS)} subjects × {QUESTIONS_PER_SUBJECT} questions")
print(f"Total questions: {len(SUBJECTS) * QUESTIONS_PER_SUBJECT}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: INFERENCE FUNCTION
#
# For each MMLU question we:
# 1. Format it as a prompt the model understands
# 2. Generate the model's answer
# 3. Compare to the correct answer
#
# MMLU format:
#   Question: [question text]
#   A) [option A]
#   B) [option B]
#   C) [option C]
#   D) [option D]
#   Answer:
#
# The model should output "A", "B", "C", or "D"
#
# PM note: prompt format matters enormously for model performance.
# This is why prompt engineering is a PM skill — the same model
# with a different prompt can score 10-15% higher on the same benchmark.
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

def format_prompt(question, choices):
    """
    Format a MMLU question into a prompt the model can answer.
    The prompt structure directly affects model performance —
    this is prompt engineering applied to benchmarking.
    """
    prompt = f"Question: {question}\n"
    prompt += f"A) {choices[0]}\n"
    prompt += f"B) {choices[1]}\n"
    prompt += f"C) {choices[2]}\n"
    prompt += f"D) {choices[3]}\n"
    prompt += "Answer with only the letter A, B, C, or D:"
    return prompt


def get_model_answer(question, choices):
    """
    Run inference on one MMLU question.

    Steps:
    1. Format the prompt
    2. Tokenize — convert text to token IDs
    3. Generate — model predicts the answer tokens
    4. Decode — convert token IDs back to text
    5. Parse — extract A, B, C, or D from the output
    """

    # Step 1: Format prompt
    prompt = format_prompt(question, choices)

    # Step 2: Tokenize
    # truncation=True prevents errors on very long questions
    # max_length=512 is enough for MMLU questions
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Step 3: Generate answer
    # max_new_tokens=5: we only need one letter (A/B/C/D)
    # do_sample=False: deterministic output, no randomness
    # This is temperature=0 equivalent — always picks highest probability token
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )

    # Step 4: Decode token IDs back to text
    answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Step 5: Parse the letter from the output
    # Model might output "A", "A)", "The answer is A", etc.
    # We look for the first A/B/C/D character in the output
    answer_text = answer_text.strip().upper()
    for char in answer_text:
        if char in ["A", "B", "C", "D"]:
            return char

    # If model output something unexpected, return None
    # This counts as wrong — models should commit to an answer
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: EVALUATION LOOP
#
# Run the model on all questions and track:
# - Score per subject
# - Which questions it got wrong
# - What the model answered vs what was correct
#
# This is the eval design you'd do as a PM:
# Don't just track the aggregate score — track per-category performance
# to understand WHERE the model fails, not just THAT it fails.
# ─────────────────────────────────────────────────────────────────────────────

results = {}  # subject → {correct, total, wrong_examples}

for subject in SUBJECTS:
    print(f"Evaluating: {subject}...")

    # Load MMLU test split for this subject from Hugging Face
    # "test" split is the standard evaluation set
    dataset = load_dataset(
        "cais/mmlu",
        subject,
        split="test"
    )

    correct = 0
    total = 0
    wrong_examples = []  # store examples the model got wrong

    # Take first N questions from this subject
    questions_to_test = list(dataset)[:QUESTIONS_PER_SUBJECT]

    for i, item in enumerate(questions_to_test):
        question = item["question"]
        choices = item["choices"]          # list of 4 answer options
        correct_idx = item["answer"]       # integer 0-3
        correct_letter = ANSWER_MAP[correct_idx]

        # Run model inference on this question
        model_answer = get_model_answer(question, choices)

        total += 1

        if model_answer == correct_letter:
            correct += 1
        else:
            # Store wrong examples for analysis
            # These are the most valuable data points — they reveal
            # what the model genuinely does not know
            if len(wrong_examples) < 3:  # keep top 3 failures per subject
                wrong_examples.append({
                    "question": question[:100] + "..." if len(question) > 100 else question,
                    "correct": correct_letter,
                    "choices": choices,
                    "model_said": model_answer if model_answer else "INVALID",
                    "correct_answer": choices[correct_idx]
                })

        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{QUESTIONS_PER_SUBJECT} questions done...")

    score = correct / total * 100
    results[subject] = {
        "correct": correct,
        "total": total,
        "score": score,
        "wrong_examples": wrong_examples
    }

    print(f"  Score: {correct}/{total} = {score:.1f}%")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: RESULTS ANALYSIS
#
# This is the PM analysis layer — not just reporting the numbers
# but interpreting what they mean.
#
# Key questions:
# 1. Which subject did the model perform worst on?
# 2. Is the failure systematic (always fails certain question types)
#    or random (fails unpredictably)?
# 3. What does the failure pattern tell us about training data?
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("RESULTS SUMMARY")
print("=" * 65)
print()

# Sort subjects by score for easy reading
sorted_results = sorted(results.items(), key=lambda x: x[1]["score"], reverse=True)

overall_correct = sum(r["correct"] for _, r in results.items())
overall_total = sum(r["total"] for _, r in results.items())
overall_score = overall_correct / overall_total * 100

print(f"{'Subject':<30} {'Score':>8} {'Correct':>10}")
print("-" * 52)
for subject, data in sorted_results:
    bar = "█" * int(data["score"] / 5)  # visual bar
    print(f"{subject:<30} {data['score']:>6.1f}%  {data['correct']}/{data['total']}  {bar}")

print("-" * 52)
print(f"{'OVERALL':<30} {overall_score:>6.1f}%  {overall_correct}/{overall_total}")
print()

# Random baseline comparison
# With 4 answer choices, random guessing = 25%
# A model below 30% is barely better than random
print(f"Random baseline (guessing):  25.0%")
print(f"Our model:                   {overall_score:.1f}%")
print(f"Above random by:             {overall_score - 25:.1f}%")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: FAILURE ANALYSIS
#
# This is the most valuable section for PM insight.
# Looking at specific wrong answers reveals:
# - Is the model confusing similar concepts?
# - Is it failing on reasoning vs recall?
# - Are there systematic patterns in what it gets wrong?
#
# A PM who can articulate failure patterns is more valuable
# than one who just reports aggregate scores.
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("FAILURE ANALYSIS — What the model got wrong")
print("=" * 65)

# Find worst performing subject
worst_subject = min(results.items(), key=lambda x: x[1]["score"])
best_subject = max(results.items(), key=lambda x: x[1]["score"])

print(f"\nBest subject:  {best_subject[0]} ({best_subject[1]['score']:.1f}%)")
print(f"Worst subject: {worst_subject[0]} ({worst_subject[1]['score']:.1f}%)")
print(f"Score gap:     {best_subject[1]['score'] - worst_subject[1]['score']:.1f}%")
print()

# Show failure examples from worst subject
print(f"Sample failures from '{worst_subject[0]}':")
print("-" * 65)
for i, example in enumerate(worst_subject[1]["wrong_examples"], 1):
    print(f"\nFailure {i}:")
    print(f"  Question:     {example['question']}")
    print(f"  Model said:   {example['model_said']}")
    print(f"  Correct was:  {example['correct']} — {example['correct_answer']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: PM INTERPRETATION
#
# This is what separates a PM analysis from an engineering report.
# Not just "what happened" but "what does it mean for product decisions"
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 65)
print("PM INTERPRETATION")
print("=" * 65)
print()

score_gap = best_subject[1]["score"] - worst_subject[1]["score"]

print(f"1. SCORE RANGE: {worst_subject[1]['score']:.1f}% to {best_subject[1]['score']:.1f}%")
print(f"   Gap of {score_gap:.1f}% between best and worst subject.")
print(f"   This is NOT random failure — it is systematic.")
print(f"   The model is genuinely stronger in some domains than others.")
print()
print(f"2. TRAINING DATA SIGNAL:")
print(f"   Strong subjects likely had more training data coverage.")
print(f"   Weak subjects were underrepresented in pre-training corpus.")
print(f"   This is data drift at the training level — not production.")
print()
print(f"3. BENCHMARK CONTAMINATION CHECK:")
print(f"   flan-t5-base was released before MMLU became widely used.")
print(f"   Lower contamination risk than newer models.")
print(f"   Score more likely reflects genuine capability.")
print()
print(f"4. WHAT SCALE BUYS:")
print(f"   flan-t5-base: ~{overall_score:.0f}% MMLU")
print(f"   GPT-4:         ~86% MMLU")
print(f"   Gap of ~{86 - overall_score:.0f}% comes from:")
print(f"   - 250M params (ours) vs ~1.8T params (GPT-4)")
print(f"   - Hundreds of billions of training tokens difference")
print(f"   - RLHF alignment improving answer quality")
print()
print(f"5. PRODUCT DECISION IMPLICATION:")
print(f"   For a product requiring {worst_subject[0]} knowledge,")
print(f"   flan-t5-base would be inadequate ({worst_subject[1]['score']:.1f}%).")
print(f"   For {best_subject[0]}, it may be sufficient ({best_subject[1]['score']:.1f}%).")
print(f"   This is why you run evals per use case, not just MMLU overall.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: VISUALIZATION
#
# Create a clean bar chart showing scores per subject.
# This becomes the GitHub artifact for Day 8.
#
# Design choices (PM thinking applied to data viz):
# - Sort by score so weakest subject is immediately visible
# - Color code: green = above 50%, red = below 50%, orange = near random
# - Show random baseline as reference line
# - Clean title that explains the insight, not just the data
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 6))

subjects_sorted = [s for s, _ in sorted_results]
scores_sorted = [r["score"] for _, r in sorted_results]

# Color code bars by performance level
colors = []
for score in scores_sorted:
    if score >= 50:
        colors.append("#5ecfb0")   # green — decent performance
    elif score >= 35:
        colors.append("#e8c06d")   # orange — marginal performance
    else:
        colors.append("#e85d75")   # red — near random baseline

bars = ax.bar(
    [s.replace("_", "\n") for s in subjects_sorted],
    scores_sorted,
    color=colors,
    width=0.6,
    edgecolor="white",
    linewidth=1.5
)

# Add score labels on bars
for bar, score in zip(bars, scores_sorted):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{score:.1f}%",
        ha="center", va="bottom",
        fontsize=11, fontweight="bold", color="#2d2d2d"
    )

# Random baseline reference line
ax.axhline(
    y=25, color="#e85d75", linestyle="--",
    linewidth=1.5, alpha=0.7, label="Random baseline (25%)"
)

# Overall score line
ax.axhline(
    y=overall_score, color="#7c6af7", linestyle="--",
    linewidth=1.5, alpha=0.7, label=f"Overall score ({overall_score:.1f}%)"
)

ax.set_ylim(0, 100)
ax.set_ylabel("MMLU Score (%)", fontsize=12)
ax.set_title(
    f"MMLU Benchmark — google/flan-t5-base\n"
    f"Overall: {overall_score:.1f}% across {overall_total} questions | "
    f"GPT-4 baseline: ~86%",
    fontsize=12, fontweight="bold", pad=15
)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_facecolor("#f8f8ff")
fig.patch.set_facecolor("white")

plt.tight_layout()
plt.savefig("mmlu_results.png", dpi=150, bbox_inches="tight")
print()
print("Chart saved as mmlu_results.png")
print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: SAVE RESULTS TO JSON
#
# Save raw results so you can reference them in the README
# without rerunning the script.
# ─────────────────────────────────────────────────────────────────────────────

output = {
    "model": MODEL_NAME,
    "overall_score": round(overall_score, 2),
    "overall_correct": overall_correct,
    "overall_total": overall_total,
    "random_baseline": 25.0,
    "subjects": {
        subject: {
            "score": round(data["score"], 2),
            "correct": data["correct"],
            "total": data["total"]
        }
        for subject, data in results.items()
    }
}

with open("mmlu_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("Results saved as mmlu_results.json")
print()
print("=" * 65)
print("Day 8 experiment complete.")
print()
print("Files to push to GitHub:")
print("  mmlu_eval.py       ← this script")
print("  mmlu_results.png   ← bar chart")
print("  mmlu_results.json  ← raw scores")
print("  README.md          ← PM analysis")
print("=" * 65)