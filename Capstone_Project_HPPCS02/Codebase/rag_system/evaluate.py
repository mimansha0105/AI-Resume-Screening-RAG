"""Evaluate candidate retrieval against human-labeled eval_data.jsonl."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Iterable
from .pipeline import ResumeRAG

def recall_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0

def precision_at_k(retrieved: list[str], relevant: set[str], k: int = 5) -> float:
    top = retrieved[:k]
    return len(set(top) & relevant) / k if k > 0 else 0.0

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, candidate_id in enumerate(retrieved, start=1):
        if candidate_id in relevant:
            return 1.0 / rank
    return 0.0

def load_cases(path: Path) -> list[dict]:
    cases = []
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        case = json.loads(line)
        if not case.get("query") or not isinstance(case.get("relevant_ids"), list):
            raise ValueError(f"Line {line_no}: expected query and relevant_ids list")
        cases.append(case)
    if not cases:
        raise ValueError("No labeled evaluation cases. Add human-reviewed rows to eval_data.jsonl.")
    return cases

def evaluate(rag: ResumeRAG, cases: Iterable[dict], k: int = 5) -> dict:
    recalls, precisions, rrs, latencies = [], [], [], []
    for case in cases:
        result = rag.retrieve(case["query"], k=k)
        retrieved = [item["resume_id"] for item in result["results"]]
        relevant = set(case["relevant_ids"])
        recalls.append(recall_at_k(retrieved, relevant, k))
        precisions.append(precision_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        latencies.append(result["retrieval_time_ms"])
    n = len(recalls)
    return {
        "queries": n,
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"precision@{k}": round(sum(precisions) / n, 4),
        "mrr": round(sum(rrs) / n, 4),
        "mean_retrieval_latency_ms": round(sum(latencies) / n, 2),
    }

def main():
    base = Path(__file__).resolve().parents[1]
    cases = load_cases(base / "eval_data.jsonl")
    rag = ResumeRAG()
    report = evaluate(rag, cases, k=5)
    print(json.dumps(report, indent=2))
    threshold = float(__import__("os").environ.get("MIN_RECALL_AT_5", "0.70"))
    actual = report["recall@5"]
    if actual < threshold:
        print(f"QUALITY GATE FAILED: Recall@5 {actual:.3f} < {threshold:.3f}", file=sys.stderr)
        sys.exit(1)
    print(f"QUALITY GATE PASSED: Recall@5 {actual:.3f} >= {threshold:.3f}")

if __name__ == "__main__":
    main()
