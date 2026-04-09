#!/usr/bin/env python3
"""RAG evaluation runner - evaluates answer quality across model/top_k configurations."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ollama import Client

from pdf_rag.config import DB_PATH, EMBED_MODEL, LLM_MODEL, OLLAMA_BASE_URL, TOP_K
from pdf_rag.llm import generate_answer
from pdf_rag.retriever import query
from pdf_rag.researcher import _open_collection

from .scorer import score


def load_dataset(path: str) -> list[dict]:
    """Load Q&A pairs from JSONL file."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_evaluation(
    dataset: list[dict],
    db_path: str,
    base_url: str,
    embed_model: str,
    models: list[str],
    top_ks: list[int],
    judge_model: str,
) -> list[dict]:
    """Run evaluation matrix and return results."""
    collection = _open_collection(db_path)
    client = Client(host=base_url)
    results = []

    total = len(models) * len(top_ks) * len(dataset)
    current = 0

    for model in models:
        for top_k in top_ks:
            for item in dataset:
                current += 1
                question = item["question"]
                ground_truth = item["ground_truth"]
                tags = item.get("tags", [])

                print(
                    f"[{current}/{total}] {model} | top_k={top_k} | {question[:50]}..."
                )

                chunks = query(question, collection, embed_model, base_url, top_k)
                if not chunks:
                    print(f"  (no chunks found)")
                    answer = ""
                    s = 0.0
                    answer_time = 0.0
                else:
                    answer_start = time.perf_counter()
                    answer = generate_answer(
                        question=question,
                        chunks=chunks,
                        base_url=base_url,
                        llm_model=model,
                        stream=False,
                    )
                    answer_time = time.perf_counter() - answer_start
                    # Score the answer (not timed)
                    s = score(
                        question=question,
                        answer=answer,
                        ground_truth=ground_truth,
                        client=client,
                        judge_model=judge_model,
                        embed_model=embed_model,
                        base_url=base_url,
                    )

                results.append(
                    {
                        "question": question,
                        "answer": answer,
                        "ground_truth": ground_truth,
                        "tags": ",".join(tags),
                        "llm_model": model,
                        "top_k": top_k,
                        "score": round(s, 4),
                        "time_seconds": round(answer_time, 2),
                    }
                )

    return results


def print_pivot_table(results: list[dict]) -> None:
    """Print pivot table: model | top_k | avg_score | avg_time | q/min."""
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Group by (model, top_k)
    groups: dict[tuple[str, int], list[dict]] = {}
    for r in results:
        key = (r["llm_model"], r["top_k"])
        groups.setdefault(key, []).append(r)

    # Print table
    print(
        f"{'Model':<20} {'Top-K':<8} {'Avg Score':<10} {'Avg Time':<10} {'Q/min':<8} {'Count':<6}"
    )
    print("-" * 80)
    for (model, top_k), items in sorted(groups.items()):
        scores = [i["score"] for i in items]
        times = [i["time_seconds"] for i in items]
        avg_score = np.mean(scores) if scores else 0.0
        avg_time = np.mean(times) if times else 0.0
        q_per_min = round(60.0 / avg_time, 1) if avg_time > 0 else 0.0
        print(
            f"{model:<20} {top_k:<8} {avg_score:<10.4f} {avg_time:<10.2f}s {q_per_min:<8.1f} {len(items):<6}"
        )
    print("=" * 80)


def save_results(results: list[dict], output_dir: str) -> Path:
    """Save results to CSV file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"eval_{ts}.csv"

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "question",
                "answer",
                "ground_truth",
                "tags",
                "llm_model",
                "top_k",
                "score",
                "time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answer quality")
    parser.add_argument(
        "--models",
        type=str,
        default=LLM_MODEL,
        help="Comma-separated model names (default: LLM_MODEL)",
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default=str(TOP_K),
        help="Comma-separated top_k values (default: TOP_K)",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default=None,
        help="Ollama model for scoring (default: first --models entry)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DB_PATH,
        help="Path to ChromaDB (default: DB_PATH)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default=OLLAMA_BASE_URL,
        help="Ollama base URL (default: OLLAMA_BASE_URL)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=EMBED_MODEL,
        help="Embedding model (default: EMBED_MODEL)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).parent / "dataset.jsonl"),
        help="Path to dataset.jsonl (default: eval/dataset.jsonl)",
    )
    parser.add_argument(
        "--num-entries",
        type=int,
        default=None,
        help="Number of dataset entries to evaluate (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).parent / "results"),
        help="Output directory for CSV (default: eval/results/)",
    )

    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    top_ks = [int(x.strip()) for x in args.top_k.split(",")]
    judge_model = args.judge if args.judge else models[0]

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)

    if args.num_entries is not None:
        dataset = dataset[: args.num_entries]

    print(f"Loaded {len(dataset)} Q&A pairs")

    print(f"\nEvaluation matrix:")
    print(f"  Models: {models}")
    print(f"  Top-K: {top_ks}")
    print(f"  Judge: {judge_model}")
    print(f"  Embed model: {args.embed_model}")
    print()

    results = run_evaluation(
        dataset=dataset,
        db_path=args.db_path,
        base_url=args.ollama_url,
        embed_model=args.embed_model,
        models=models,
        top_ks=top_ks,
        judge_model=judge_model,
    )

    output_path = save_results(results, args.output_dir)
    print(f"\nResults saved to: {output_path}")

    print_pivot_table(results)


if __name__ == "__main__":
    main()
