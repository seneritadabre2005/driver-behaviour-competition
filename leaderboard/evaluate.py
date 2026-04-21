import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate(submission_path: str, ground_truth_path: str) -> dict:
    gt = pd.read_csv(ground_truth_path)
    sub = pd.read_csv(submission_path)

    # If no row_id, create one from the dataframe index
    if "row_id" not in sub.columns:
        sub = sub.reset_index()
        sub.columns = ["row_id"] + [c for c in sub.columns if c != "row_id"][1:]
        sub["row_id"] = range(len(sub))

    assert "Class" in sub.columns, "Submission must have a 'Class' column."
    assert len(sub) == len(gt), f"Row count mismatch: expected {len(gt)}, got {len(sub)}."

    y_true = gt["Class"].values
    y_pred = sub["Class"].values

    accuracy = round(accuracy_score(y_true, y_pred), 4)
    f1 = round(f1_score(y_true, y_pred, average="weighted"), 4)

    return {"accuracy": accuracy, "f1_weighted": f1}