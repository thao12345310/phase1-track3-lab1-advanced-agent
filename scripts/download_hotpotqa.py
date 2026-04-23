"""
Download HotpotQA data from HuggingFace and convert to the lab's JSON format.
No `datasets` library needed — uses only `requests` + stdlib.

Usage:
    python scripts/download_hotpotqa.py --num 120 --out data/hotpot_full.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import requests

# HuggingFace dataset API for HotpotQA (distractor setting, validation split)
HF_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "validation"
MAX_PER_REQUEST = 100  # HF API limit per request


def fetch_rows(offset: int, length: int) -> list[dict]:
    """Fetch rows from HuggingFace datasets server API."""
    resp = requests.get(
        HF_API_URL,
        params={
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": min(length, MAX_PER_REQUEST),
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [row["row"] for row in data.get("rows", [])]


def classify_difficulty(level: str) -> str:
    """Map HotpotQA level to our difficulty enum."""
    level = (level or "").lower().strip()
    if level == "easy":
        return "easy"
    elif level == "medium":
        return "medium"
    else:
        return "hard"


def convert_row(row: dict, idx: int) -> dict | None:
    """Convert a HotpotQA row to our QAExample JSON format."""
    question = row.get("question", "").strip()
    answer = row.get("answer", "").strip()
    level = row.get("level", "medium")

    if not question or not answer:
        return None

    # Build context from supporting facts
    titles = row.get("context", {}).get("title", [])
    sentences_list = row.get("context", {}).get("sentences", [])

    context_chunks = []
    seen_titles = set()
    for title, sentences in zip(titles, sentences_list):
        if title in seen_titles:
            continue
        seen_titles.add(title)
        text = " ".join(sentences).strip()
        if text:
            context_chunks.append({"title": title, "text": text})

    # Need at least 2 context chunks for multi-hop
    if len(context_chunks) < 2:
        return None

    # Limit to first 4 context chunks to keep prompts manageable
    context_chunks = context_chunks[:4]

    return {
        "qid": f"hq{idx}",
        "difficulty": classify_difficulty(level),
        "question": question,
        "gold_answer": answer,
        "context": context_chunks,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download HotpotQA data")
    parser.add_argument("--num", type=int, default=120,
                        help="Number of examples to download (default: 120)")
    parser.add_argument("--out", type=str, default="data/hotpot_full.json",
                        help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    target = args.num
    out_path = Path(args.out)

    print(f"Downloading {target} HotpotQA examples...")
    print(f"Source: {DATASET} ({CONFIG}/{SPLIT})")

    # Fetch more than needed to account for filtered-out rows
    fetch_total = min(target * 2, 500)
    all_rows = []

    for offset in range(0, fetch_total, MAX_PER_REQUEST):
        batch_size = min(MAX_PER_REQUEST, fetch_total - offset)
        print(f"  Fetching rows {offset}..{offset + batch_size}...")
        try:
            rows = fetch_rows(offset, batch_size)
            all_rows.extend(rows)
        except Exception as e:
            print(f"  ⚠ Error at offset {offset}: {e}")
            break

    print(f"  Fetched {len(all_rows)} raw rows")

    # Convert and filter
    examples = []
    for i, row in enumerate(all_rows):
        ex = convert_row(row, i + 1)
        if ex:
            examples.append(ex)

    print(f"  Converted {len(examples)} valid examples")

    # Shuffle and trim to target
    random.seed(args.seed)
    random.shuffle(examples)
    examples = examples[:target]

    # Re-index qids
    for i, ex in enumerate(examples, 1):
        ex["qid"] = f"hq{i}"

    # Ensure difficulty distribution
    diff_counts = {}
    for ex in examples:
        diff_counts[ex["difficulty"]] = diff_counts.get(ex["difficulty"], 0) + 1
    print(f"  Difficulty distribution: {diff_counts}")

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ Saved {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()
