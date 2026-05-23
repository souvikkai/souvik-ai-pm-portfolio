"""
Run benchmark prompts against Anthropic Claude and save results.
Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import time
from pathlib import Path

from anthropic import Anthropic

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

PROMPTS_FILE = PROJECT_ROOT / "prompts" / "benchmark_prompts.json"
OUTPUT_FILE = PROJECT_ROOT / "outputs" / "claude_results.json"

# --- Model settings ---
MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 1024


def load_prompts():
    """Load benchmark prompts from JSON file."""
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def ask_claude(prompt_text, client):
    """
    Send prompt to Claude and return:
    - response text
    - latency in seconds
    """

    start = time.perf_counter()

    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": prompt_text
            }
        ]
    )

    latency_seconds = time.perf_counter() - start

    response_text = message.content[0].text

    return response_text, latency_seconds


def main():

    # Read API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        raise SystemExit(
            "Error: Set the ANTHROPIC_API_KEY environment variable."
        )

    # Create Claude client
    client = Anthropic(api_key=api_key)

    # Load prompts
    prompts = load_prompts()

    results = []

    print(f"Running {len(prompts)} prompts with {MODEL}...\n")

    # Run benchmark
    for item in prompts:

        prompt_id = item["id"]
        category = item["category"]
        prompt = item["prompt"]

        print(
            f"  [{prompt_id}] {category} ... ",
            end="",
            flush=True
        )

        response, latency_seconds = ask_claude(prompt, client)

        results.append({
            "prompt_id": prompt_id,
            "category": category,
            "prompt": prompt,
            "response": response,
            "latency_seconds": round(latency_seconds, 3),
        })

        print(f"{latency_seconds:.2f}s")

    # Save outputs
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()