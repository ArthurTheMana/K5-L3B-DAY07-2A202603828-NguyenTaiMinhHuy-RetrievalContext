
import json
import re
import unicodedata
import difflib
import csv
from pathlib import Path

data = Path("data/converted")
chunks = json.loads((data / "chunks.json").read_text(encoding="utf-8-sig"))
questions = json.loads((data / "questions.json").read_text(encoding="utf-8-sig"))

def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()

rows = []

for q in questions:
    if q["split"] != "dev":
        continue

    for gold in q["gold"]:
        matches = [
            c for c in chunks
            if c["source_id"] == gold["source_id"]
            and c["page"] == gold["page"]
        ]

        evidence = normalize(gold["evidence"])

        exact = any(gold["evidence"] in c["text"] for c in matches)
        normalized = any(evidence in normalize(c["text"]) for c in matches)

        best_score = 0
        best_chunk = None

        for chunk in matches:
            score = difflib.SequenceMatcher(
                None, evidence, normalize(chunk["text"]),
                autojunk=False
            ).find_longest_match().size / max(len(evidence), 1)

            if score > best_score:
                best_score = score
                best_chunk = chunk

        status = (
            "EXACT" if exact else
            "NORMALIZED" if normalized else
            "NO_SOURCE_PAGE" if not matches else
            "NEEDS_REVIEW"
        )

        rows.append({
            "question_id": q["id"],
            "status": status,
            "best_match_percent": round(best_score * 100, 1),
            "best_chunk_id": best_chunk["id"] if best_chunk else "",
            "gold_evidence": gold["evidence"],
            "chunk_text": best_chunk["text"] if best_chunk else ""
        })

        print(q["id"], status, round(best_score * 100, 1))

output = Path("results/dev_gold_audit.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Audit completed:", output)
