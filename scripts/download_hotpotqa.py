"""
Download HotpotQA data split by difficulty level (easy / medium / hard).
Fetches from HuggingFace API — no `datasets` library needed.

Output:
    data/hotpot_easy.json   (~20 examples)
    data/hotpot_medium.json (~20 examples)
    data/hotpot_hard.json   (~20 examples)
    data/hotpot_full.json   (all combined, 60+ examples)

Usage:
    python scripts/download_hotpotqa.py
    python scripts/download_hotpotqa.py --per-level 40
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# HuggingFace Datasets Server API
# ---------------------------------------------------------------------------
HF_API_URL = "https://datasets-server.huggingface.co/rows"
DATASET = "hotpotqa/hotpot_qa"
CONFIG = "distractor"
SPLIT = "train"  # train split has easy/medium/hard mix
MAX_PER_REQUEST = 100


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
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [row["row"] for row in data.get("rows", [])]


def classify_difficulty(level: str) -> str:
    """Map HotpotQA level to difficulty."""
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
    level = row.get("level", "hard")

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

    if len(context_chunks) < 2:
        return None

    # Limit to first 4 chunks
    context_chunks = context_chunks[:4]

    return {
        "qid": "",  # will be assigned later
        "difficulty": classify_difficulty(level),
        "question": question,
        "gold_answer": answer,
        "context": context_chunks,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download HotpotQA data by difficulty")
    parser.add_argument("--per-level", type=int, default=20,
                        help="Number of examples per difficulty level (default: 20)")
    parser.add_argument("--out-dir", type=str, default="data",
                        help="Output directory (default: data)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    target_per_level = args.per_level
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎯 Target: {target_per_level} examples per difficulty level")
    print(f"📦 Source: {DATASET} ({CONFIG}/{SPLIT})\n")

    # Collect examples by difficulty
    by_diff: dict[str, list[dict]] = defaultdict(list)
    offset = 0
    total_fetched = 0
    max_fetch = 1000  # safety limit

    while offset < max_fetch:
        # Check if we have enough of each
        have_enough = all(
            len(by_diff[d]) >= target_per_level
            for d in ("easy", "medium", "hard")
        )
        if have_enough:
            break

        batch_size = MAX_PER_REQUEST
        print(f"  Fetching rows {offset}..{offset + batch_size}... ", end="")
        try:
            rows = fetch_rows(offset, batch_size)
            if not rows:
                print("no more rows")
                break
        except Exception as e:
            print(f"⚠ Error: {e}")
            break

        batch_converted = 0
        for row in rows:
            ex = convert_row(row, 0)
            if ex:
                by_diff[ex["difficulty"]].append(ex)
                batch_converted += 1

        total_fetched += len(rows)
        counts = {d: len(by_diff[d]) for d in ("easy", "medium", "hard")}
        print(f"got {batch_converted} valid → easy={counts['easy']}, medium={counts['medium']}, hard={counts['hard']}")

        offset += batch_size

    print(f"\n📊 Total rows fetched: {total_fetched}")

    # Shuffle and trim each level
    random.seed(args.seed)
    all_combined = []

    for difficulty in ("easy", "medium", "hard"):
        examples = by_diff[difficulty]
        random.shuffle(examples)
        examples = examples[:target_per_level]

        # Assign qids
        prefix = {"easy": "he", "medium": "hm", "hard": "hh"}[difficulty]
        for i, ex in enumerate(examples, 1):
            ex["qid"] = f"{prefix}{i}"

        # Save individual file
        file_path = out_dir / f"hotpot_{difficulty}.json"
        file_path.write_text(
            json.dumps(examples, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  ✅ Saved {len(examples):3d} {difficulty:6s} examples → {file_path}")
        all_combined.extend(examples)

    # Re-index and save combined file
    random.shuffle(all_combined)
    for i, ex in enumerate(all_combined, 1):
        ex["qid"] = f"hq{i}"

    combined_path = out_dir / "hotpot_full.json"
    combined_path.write_text(
        json.dumps(all_combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    diff_counts = defaultdict(int)
    for ex in all_combined:
        diff_counts[ex["difficulty"]] += 1
    print(f"  ✅ Saved {len(all_combined):3d} combined examples → {combined_path}")
    print(f"\n📈 Final distribution: {dict(diff_counts)}")
    print(f"📁 Total examples: {len(all_combined)}")


if __name__ == "__main__":
    main()
