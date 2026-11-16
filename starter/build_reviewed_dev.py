
import json
from pathlib import Path

ROOT = Path("data")
SOURCE = ROOT / "converted"
chunks = json.loads(
    (SOURCE / "chunks.json").read_text(encoding="utf-8-sig")
)
questions = json.loads(
    (SOURCE / "questions.json").read_text(encoding="utf-8-sig")
)

by_id = {c["id"]: c for c in chunks}

# Each entry: chunk ID, starting text, ending text.
# Multiple entries mean multiple pieces of required evidence.
EVIDENCE = {
    "Q01": [
        ("uow_ai_ml_p005_paragraph_001",
         "technologies working together",
         "human-like levels of intelligence.")
    ],
    "Q02": [
        ("uow_ai_ml_p010_paragraph_001",
         "Strong AI aims to create",
         "such as answering questions based on user input or playing chess.")
    ],
    "Q03": [
        ("uow_ai_ml_p015_paragraph_001",
         "Six phases",
         "Operationalize (delivery, pilot project)")
    ],
    "Q04": [
        ("uow_ai_ml_p019_paragraph_001",
         "Feature selection :",
         "Creating new features by gathering new data")
    ],
    "Q05": [
        ("uow_ai_ml_p020_paragraph_001",
         "In this learning scenario",
         "example data.")
    ],
    "Q06": [
        ("uow_ai_ml_p021_paragraph_001",
         "Separate data into three distinct subsets.",
         "training, test, and validation dataset.")
    ],
    "Q07": [
        ("uow_ai_ml_p025_paragraph_001",
         "In this learning scenario",
         "We want to uncover the pattern inherent in the data.")
    ],
    "Q08": [
        ("uow_ai_ml_p030_paragraph_001",
         "In this learning scenario",
         "many unlabelled data.")
    ],
    "Q11": [
        ("colliot_ml_2023_p011_paragraph_001",
         "Deep learning is a type",
         "other approaches\nexist.")
    ],
    "Q12": [
        ("colliot_ml_2023_p012_paragraph_001",
         "a computer program is said to learn",
         "experience E.")
    ],
    "Q13": [
        ("colliot_ml_2023_p013_paragraph_001",
         "Learns from labelled examples",
         "known."),
        ("colliot_ml_2023_p013_paragraph_002",
         "Unsupervised learning. Learns from unlabelled examples.",
         "Unsupervised learning. Learns from unlabelled examples.")
    ],
    "Q14": [
        ("colliot_ml_2023_p015_paragraph_001",
         "In unsupervised learning, the examples are not labeled.",
         "common tasks in unsupervised learning a"),
        ("colliot_ml_2023_p015_paragraph_002",
         "re clustering and dimensionality",
         "reduction (Figure 8).")
    ],
    "Q15": [
        ("colliot_ml_2023_p017_paragraph_001",
         "self-supervised learning, where",
         "vision, has emerged.")
    ],
    "Q16": [
        ("colliot_ml_2023_p019_paragraph_001",
         "minimizes the average error over the training set.",
         "error the cost function:")
    ],
    "Q17": [
        ("colliot_ml_2023_p019_paragraph_004",
         "the learned function f may be too",
         "more general rule that would work for other examples.")
    ],
}

dev = [q for q in questions if q["split"] == "dev"]

assert set(EVIDENCE) == {q["id"] for q in dev}

for q in dev:
    original_gold = q["gold"]
    updated = []

    for chunk_id, start, end in EVIDENCE[q["id"]]:
        chunk = by_id[chunk_id]
        text = chunk["text"]

        begin = text.find(start)
        if begin < 0:
            raise ValueError(f"{q['id']}: Start text not found")

        finish = text.find(end, begin)
        if finish < 0:
            raise ValueError(f"{q['id']}: End text not found")

        evidence = text[begin:finish + len(end)]

        assert chunk["source_id"] == original_gold[0]["source_id"]
        assert chunk["page"] == original_gold[0]["page"]
        assert evidence in chunk["text"]

        updated.append({
            "source_id": chunk["source_id"],
            "page": chunk["page"],
            "evidence": evidence
        })

    q["gold"] = updated
    print(q["id"], "OK", len(updated), "evidence piece(s)")

output = ROOT / "benchmark_dev_reviewed.json"
output.write_text(
    json.dumps(dev, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8"
)

print("Created:", output)
print("Dev questions:", len(dev))
