
import argparse
import json
import statistics
import time
from pathlib import Path

import os
import hashlib
import psycopg

import numpy as np
from sentence_transformers import SentenceTransformer

from baseline import validate, hit, pack_context

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def complete_evidence(chunks, gold):
    return all(
        any(hit(c, [g]) for c in chunks)
        for g in gold
    )


def load_vectors_from_pg(chunks):
    expected = {
        c["id"]: hashlib.sha256(
            c["text"].encode("utf-8")
        ).hexdigest()
        for c in chunks
    }

    with psycopg.connect(
        host="127.0.0.1",
        port=55433,
        dbname=os.environ["DAY7_DB_NAME"],
        user=os.environ["DAY7_DB_USER"],
        password=os.environ["DAY7_DB_PASSWORD"]
    ) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_id, text_sha256, embedding::text
                FROM public.day7_minilm_chunks
            """)
            stored = {
                chunk_id: (digest, vector)
                for chunk_id, digest, vector in cur.fetchall()
            }

    if set(stored) != set(expected):
        raise ValueError("Chunk IDs do not match PostgreSQL")

    for chunk_id, digest in expected.items():
        if stored[chunk_id][0] != digest:
            raise ValueError(
                f"Text mismatch for chunk: {chunk_id}"
            )

    vectors = np.stack([
        np.fromstring(
            stored[c["id"]][1].strip("[]"),
            sep=",",
            dtype=np.float32
        )
        for c in chunks
    ])

    if vectors.shape != (len(chunks), 384):
        raise ValueError("Invalid embedding dimensions")

    print("Loaded vectors from PostgreSQL:", len(vectors))
    return vectors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    chunks = json.loads(
        args.chunks.read_text(encoding="utf-8-sig")
    )
    questions = json.loads(
        args.questions.read_text(encoding="utf-8-sig")
    )
    dev = [q for q in questions if q["split"] == "dev"]

    validate(chunks, dev)

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL)

    print("Loading vectors from PostgreSQL...")
    start = time.perf_counter()
    vectors = load_vectors_from_pg(chunks)
    index_seconds = time.perf_counter() - start

    traces = []

    for q in dev:
        start = time.perf_counter()

        query_vector = model.encode(
            q["question"],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores = np.dot(vectors, query_vector)

        ranked = sorted(
            range(len(chunks)),
            key=lambda i: (-float(scores[i]), chunks[i]["id"])
        )[:3]

        results = [
            {
                **chunks[i],
                "score": float(scores[i]),
                "strategy": "dense"
            }
            for i in ranked
        ]

        context, context_ids = pack_context(
            results, budget=800
        )

        marked = [
            {**c, "hit": hit(c, q["gold"])}
            for c in results
        ]

        context_chunks = [
            c for c in results if c["id"] in context_ids
        ]

        traces.append({
            "query_id": q["id"],
            "split": "dev",
            "question": q["question"],
            "strategy": "dense",
            "top_k": 3,
            "budget_chars": 800,
            "gold": q["gold"],
            "results": marked,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": (
                time.perf_counter() - start
            ) * 1000,
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
        "strategy": "dense",
        "model": MODEL,
        "split": "dev",
        "n": len(traces),
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

    for name, obj in (
        ("traces.json", traces),
        ("summary.json", summary)
    ):
        (args.output / name).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
