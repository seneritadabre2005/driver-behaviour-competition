# Driving Behaviour Anomaly Detection — ML Competition

## Task
Classify driving sessions into: **Normal**, **Aggressive**, or **Drowsy**

## Dataset
- `data/train_motion_data.csv` — use this to train your model
- `data/test_motion_data_nolabels.csv` — generate predictions for this file

## Baseline
Run `baseline/baseline_model.py` to see the baseline Random Forest model and generate a sample submission.

## Submission Format
Your file must be named: `TeamName_submission.csv`  
Required columns:

| row_id | Class |
|--------|-------|
| 0 | Normal |
| 1 | Aggressive |

- `Class` must be exactly: `Normal`, `Aggressive`, or `Drowsy`
- Submit via Pull Request into the `submissions/` folder

## Evaluation
Teams are ranked by **Weighted F1 Score**, then Accuracy.

## Leaderboard
[View Live Leaderboard](https://siyaagarwal2005.github.io/driver-behaviour-competition/)