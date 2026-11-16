
import argparse
import json
import statistics
import time
from pathlib import Path

from dense_retrieval import load_vectors_from_pg

import numpy as np
from sentence_transformers import SentenceTransformer

from baseline import TfidfIndex, validate, hit, pack_context

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
CANDIDATES = 20
CONTEXT_BUDGET = 1200


def complete_evidence(chunks, gold):
    return all(
        any(hit(chunk, [g]) for chunk in chunks)
        for g in gold
    )


def rrf_fusion(sparse_results, dense_results):
    scores = {}

    for ranked_list in (sparse_results, dense_results):
        for rank, chunk in enumerate(ranked_list, start=1):
            chunk_id = chunk["id"]
            scores[chunk_id] = (
                scores.get(chunk_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    return scores





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    chunks = json.loads(            
        args.chunks.read_text(encoding="utf-8-sig")
    )
    questions = json.loads(
        args.questions.read_text(encoding="utf-8-sig")
    )

    dev = [q for q in questions if q["split"] == args.split]
    validate(chunks, dev)

    print("Building TF-IDF index...")
    start = time.perf_counter()
    tfidf = TfidfIndex(chunks)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL)

    print("Loading vectors from PostgreSQL...")
    vectors = load_vectors_from_pg(chunks)

    index_seconds = time.perf_counter() - start
    by_id = {c["id"]: c for c in chunks}

    traces = []

    for q in dev:
        start = time.perf_counter()

        sparse = tfidf.search(
            q["question"], top_k=CANDIDATES
        )

        query_vector = model.encode(
            q["question"],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        dense_scores = np.dot(vectors, query_vector)

        dense_ranks = sorted(
            range(len(chunks)),
            key=lambda i: (
                -float(dense_scores[i]),
                chunks[i]["id"]
            )
        )[:CANDIDATES]

        dense = [chunks[i] for i in dense_ranks]

        fused = rrf_fusion(sparse, dense)

        ranked_ids = sorted(
            fused,
            key=lambda chunk_id: (
                -fused[chunk_id], chunk_id
            )
        )[:3]

        results = [
            {
                **by_id[chunk_id],
                "score": fused[chunk_id],
                "strategy": "hybrid_rrf"
            }
            for chunk_id in ranked_ids
        ]

        context, context_ids = pack_context(
            results, budget=CONTEXT_BUDGET
        )

        context_chunks = [
            c for c in results
            if c["id"] in context_ids
        ]

        marked = [
            {**c, "hit": hit(c, q["gold"])}
            for c in results
        ]

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        traces.append({
            "query_id": q["id"],
            "split": args.split,
            "question": q["question"],
            "strategy": "hybrid_rrf",
            "top_k": 3,
            "budget_chars": CONTEXT_BUDGET,
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": complete_evidence(
                results[:1], q["gold"]
            ),
            "hit3": complete_evidence(
                results, q["gold"]
            ),
            "context_hit": complete_evidence(
                context_chunks, q["gold"]
            )
        })

    summary = {
        "strategy": "hybrid_rrf",
        "model": MODEL,
        "split": args.split,
        "n": len(traces),
        "rrf_k": RRF_K,
        "candidates_per_method": CANDIDATES,
        "context_budget_chars": CONTEXT_BUDGET,
        "hit1": sum(t["hit1"] for t in traces),
        "hit3": sum(t["hit3"] for t in traces),
        "context_hit": sum(
            t["context_hit"] for t in traces
        ),
        "avg_context_chars": statistics.mean(
            t["context_chars"] for t in traces
        ),
        "median_ms": statistics.median(
            t["elapsed_ms"] for t in traces
        ),
        "index_seconds": index_seconds,
        "service_cost_usd": 0
    }

    args.output.mkdir(parents=True, exist_ok=True)

    for filename, data in (
        ("summary.json", summary),
        ("traces.json", traces)
    ):
        (args.output / filename).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
