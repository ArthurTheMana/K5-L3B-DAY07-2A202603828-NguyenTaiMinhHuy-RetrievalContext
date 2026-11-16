
import csv
from pathlib import Path

questions = [
    (
        "U01", "dev",
        "What exact test accuracy did the UOW lecture obtain from "
        "training a Random Forest classifier on the MNIST dataset?"
    ),
    (
        "U02", "dev",
        "What exact training time in seconds did Colliot report "
        "for a deep learning model using an NVIDIA GPU?"
    ),
    (
        "U03", "test",
        "What exact F1-score did the UOW lecture measure for a "
        "supervised learning model on a customer churn dataset?"
    ),
    (
        "U04", "test",
        "What exact batch size and number of training epochs were "
        "used in Colliot's reported medical-image classification experiment?"
    ),
]

output = Path("results/unanswerable_review.csv")
output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "split", "question",
        "verification_method", "verification_status", "review_notes"
    ])

    for qid, split, question in questions:
        writer.writerow([
            qid,
            split,
            question,
            "Review both source PDFs and their relevant pages",
            "PENDING",
            ""
        ])

print("Created:", output)
print("Candidate questions:", len(questions))
print("Dev:", 2)
print("Test:", 2)
print("Status: Pending manual verification")
