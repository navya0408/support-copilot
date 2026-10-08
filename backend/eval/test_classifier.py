"""Check the classifier on 21 hand-written tickets (a different style from the training data).

Run from the backend/ folder:   python -m eval.test_classifier

This is a small STRESS TEST, not an official score: the labels are my own judgment and
some tickets are genuinely ambiguous. The official number is the test-set score printed
by ml/train_baseline.py.
"""
import json
from collections import Counter
from pathlib import Path

from app import classify

HERE = Path(__file__).parent


def main():
    items = json.loads((HERE / "classifier_samples.json").read_text(encoding="utf-8"))
    correct = 0
    confident = [0, 0]      # [correct, total] for confidence >= 0.5
    unsure = [0, 0]         # [correct, total] for confidence <  0.5
    confusions = Counter()

    for it in items:
        pred, conf, top3 = classify.predict(it["text"])
        ok = pred == it["label"]
        correct += ok
        bucket = confident if conf >= 0.5 else unsure
        bucket[0] += ok
        bucket[1] += 1
        if not ok:
            confusions[(it["label"], pred)] += 1
        print(f"{'OK' if ok else 'XX'}  conf={conf:.2f}  true={it['label']:<22} pred={pred:<22} | {it['text'][:60]}")

    n = len(items)
    print(f"\nAccuracy on hand-written tickets: {correct}/{n} = {correct / n:.0%}")
    if confident[1]:
        print(f"  confidence >= 0.5: {confident[0]}/{confident[1]} correct")
    if unsure[1]:
        print(f"  confidence <  0.5: {unsure[0]}/{unsure[1]} correct  (the UI flags these as low confidence)")
    if confusions:
        print("\nMost common mix-ups (true -> predicted):")
        for (t, p), c in confusions.most_common():
            print(f"  {t} -> {p}: {c}")


if __name__ == "__main__":
    main()
