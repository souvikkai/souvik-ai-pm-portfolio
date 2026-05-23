"""
Run benchmark prompts against local Ollama (llama3.1:8b).
Saves latency and responses to outputs/llama_results.json.
"""

import json
import time
from pathlib import Path
from urllib import request

# --- Paths (relative to this script) ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PROMPTS_FILE = PROJECT_DIR / "prompts" / "benchmark_prompts.json"
OUTPUT_FILE = PROJECT_DIR / "outputs" / "llama_results.json"

# --- Ollama settings ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"


def load_prompts():
    """Load the list of benchmark prompts from JSON."""
    with open(PROMPTS_FILE, encoding="utf-8") as f:
        return json.load(f)


def query_ollama(prompt_text):
    """
    Send one prompt to Ollama and return (response_text, latency_seconds).
    Uses the /api/generate endpoint with streaming disabled.
    """
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt_text,
        "stream": False,
    }).encode("utf-8")

    req = request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - start

    return data.get("response", ""), latency


def main():
    # Load all prompts from the benchmark file
    prompts = load_prompts()
    results = []

    # Run each prompt and collect timing + response
    for item in prompts:
        print(f"Running prompt {item['id']} ({item['category']})...")

        response, latency = query_ollama(item["prompt"])

        results.append({
            "prompt_id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "response": response,
            "latency_seconds": round(latency, 3),
        })

    # Write results to outputs/llama_results.json
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Done. Saved {len(results)} results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
