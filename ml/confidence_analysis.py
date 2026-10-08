"""How trustworthy is the classifier's confidence?  Choose the "low confidence" threshold.

Put this file in the ml/ folder and run from the project ROOT:   python ml/confidence_analysis.py

It uses the VALIDATION set only (never the test set) to pick the threshold.
For each threshold t, tickets with confidence < t would be flagged "check manually".
"""
import joblib
import numpy as np

from common import ROOT, load_splits

model = joblib.load(ROOT / "backend" / "models" / "ticket_classifier.joblib")
_, val, _ = load_splits()

proba = model.predict_proba(val["text"])
pred = model.classes_[proba.argmax(axis=1)]
conf = proba.max(axis=1)
correct = pred == val["label"].to_numpy()

print(f"Validation tickets: {len(val)}   overall accuracy: {correct.mean():.3f}\n")
print("Accuracy by confidence band:")
bands = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
for lo, hi in zip(bands[:-1], bands[1:]):
    m = (conf >= lo) & (conf < hi)
    if m.sum():
        print(f"  {lo:.1f}-{min(hi, 1.0):.1f}   n={m.sum():<5} accuracy={correct[m].mean():.3f}")

print("\nIf we flag every ticket below the threshold as 'check manually':")
print(f"  {'threshold':<10}{'flagged':<10}{'accuracy of unflagged':<24}{'errors caught'}")
total_errors = (~correct).sum()
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    flagged = conf < t
    keep = ~flagged
    acc_keep = correct[keep].mean() if keep.any() else float("nan")
    caught = (flagged & ~correct).sum() / total_errors if total_errors else 0.0
    print(f"  {t:<10}{flagged.mean():<10.1%}{acc_keep:<24.3f}{caught:.1%}")
print("\nPick the threshold where unflagged accuracy is high but you are not flagging most tickets.")