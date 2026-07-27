import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cerpt.data.synthetic import generate_dataset, make_adversarial_rows
from cerpt.verification.audit import ablate_input
from cerpt.verification.deterministic import check_example


def test_deterministic_checker_accepts_generated_data():
    rows = generate_dataset(32, seed=123)
    assert all(check_example(row)["valid"] for row in rows)


def test_adversarial_rows_are_labeled_invalid():
    rows = make_adversarial_rows(generate_dataset(8, seed=123), seed=2)
    assert len(rows) == 8
    assert all(row["evidence_label"] == 0 for row in rows)
    assert all(any(not item["valid"] for item in row["trace"]) for row in rows)


def test_ablation_changes_the_prompt():
    row = generate_dataset(1, seed=123)[0]
    assert ablate_input(row) != row["input_text"]
