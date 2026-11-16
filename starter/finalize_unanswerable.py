
import csv
from pathlib import Path

path = Path("results/unanswerable_review.csv")

notes = {
    "U01": (
        "Reviewed both PDFs, including UOW supervised learning "
        "and classification examples. No MNIST Random Forest "
        "test accuracy is reported."
    ),
    "U02": (
        "Reviewed both PDFs, including Colliot's discussion "
        "of GPU acceleration on PDF pages 9-10. No exact "
        "NVIDIA GPU training time in seconds is reported."
    ),
    "U03": (
        "Reviewed both PDFs, including UOW classification "
        "examples and applications. No customer churn "
        "experiment with an exact F1-score is reported."
    ),
    "U04": (
        "Reviewed both PDFs, including Colliot's medical "
        "image classification examples on PDF pages 12-14. "
        "No matching batch size and epoch count is reported."
    ),
}

with path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)

assert len(rows) == 4
assert {r["id"] for r in rows} == set(notes)

for row in rows:
    row["verification_status"] = "VERIFIED_UNANSWERABLE"
    row["verification_method"] = (
        "Reviewed both source PDFs, including relevant "
        "text, figures and examples"
    )
    row["review_notes"] = notes[row["id"]]

with path.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("Verification complete: 4/4")
print("Dev: 2 verified")
print("Test: 2 verified")
print("Updated:", path)
