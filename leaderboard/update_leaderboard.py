import os
import json
import pandas as pd
from evaluate import evaluate

SUBMISSIONS_DIR = "../submissions/"
GROUND_TRUTH = "../data/test_motion_data.csv"   # This stays LOCAL, never pushed
LEADERBOARD_FILE = "leaderboard.json"

def update():
    results = []

    for fname in os.listdir(SUBMISSIONS_DIR):
        if not fname.endswith(".csv"):
            continue

        team_name = fname.replace("_submission.csv", "")
        path = os.path.join(SUBMISSIONS_DIR, fname)

        try:
            metrics = evaluate(path, GROUND_TRUTH)
            results.append({
                "team": team_name,
                "accuracy": metrics["accuracy"],
                "f1_weighted": metrics["f1_weighted"]
            })
            print(f"✓ Scored: {team_name}")
        except Exception as e:
            print(f"✗ Failed for {team_name}: {e}")

    # Sort by F1, then accuracy
    results.sort(key=lambda x: (x["f1_weighted"], x["accuracy"]), reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1

    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nLeaderboard updated. {len(results)} team(s) ranked.")

if __name__ == "__main__":
    update()