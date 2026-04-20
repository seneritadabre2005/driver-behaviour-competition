import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate(submission_path: str, ground_truth_path: str) -> dict:
    """
    Evaluates a team's submission against ground truth.
    Returns a dict of metrics.
    """
    gt = pd.read_csv(ground_truth_path)
    sub = pd.read_csv(submission_path)

    # Validate
    assert "row_id" in sub.columns and "Class" in sub.columns, \
        "Submission must have 'row_id' and 'Class' columns."
    assert len(sub) == len(gt), \
        f"Row count mismatch: expected {len(gt)}, got {len(sub)}."

    # Merge on row_id to handle ordering
    merged = gt.merge(sub, on="row_id", suffixes=("_true", "_pred"))

    y_true = merged["Class_true"]
    y_pred = merged["Class_pred"]

    accuracy = round(accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred, average="weighted"), 4)

    return {"accuracy": accuracy, "f1_weighted": f1}