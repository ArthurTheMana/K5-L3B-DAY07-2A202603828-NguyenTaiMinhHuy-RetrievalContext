
import json
import csv
import unicodedata
import re
from pathlib import Path

root = Path("data/converted")

chunks = json.loads(
    (root / "chunks.json").read_text(encoding="utf-8-sig")
)
questions = json.loads(
    (root / "questions.json").read_text(encoding="utf-8-sig")
)

def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()

review = []

for q in questions:
    if q["split"] != "dev":
        continue

    for gold in q["gold"]:
        matching = [
            c for c in chunks
            if c["source_id"] == gold["source_id"]
            and c["page"] == gold["page"]
        ]

        valid = [
            c for c in matching
            if normalize(gold["evidence"]) in normalize(c["text"])
        ]

        review.append({
            "question_id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "source_id": gold["source_id"],
            "page": gold["page"],
            "status": "NORMALIZED_MATCH" if valid else "NEEDS_REVIEW",
            "matching_chunk_ids": " | ".join(c["id"] for c in matching),
            "gold_evidence": gold["evidence"],
            "matching_chunk_texts": "\n\n".join(
                f"[{c['id']}]\n{c['text']}" for c in matching
            )
        })

output = Path("results/dev_evidence_review.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=review[0].keys())
    writer.writeheader()
    writer.writerows(review)

print("Review file:", output)
print("Questions:", len(review))
print("Need review:", sum(r["status"] == "NEEDS_REVIEW" for r in review))
