"""Train the TF-IDF + Logistic Regression ticket classifier (the model we deploy).

Input : data/cfpb_clean.csv
Output: backend/models/ticket_classifier.joblib, ml/results/baseline_metrics.json

Run:  python ml/train_baseline.py

The TEST set is evaluated exactly once, at the end. Do not tune on it.
"""
import json
from datetime import date

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

from common import ROOT, RESULTS_DIR, load_splits

MODEL_OUT = ROOT / "backend" / "models" / "ticket_classifier.joblib"


def scores(y_true, y_pred):
    return {
        "accuracy": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def main():
    train, val, test = load_splits()
    print("split sizes:", len(train), len(val), len(test))

    dummy = DummyClassifier(strategy="most_frequent").fit(train["text"], train["label"])
    results = {"majority_class_val": scores(val["label"], dummy.predict(val["text"]))}

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                                  sublinear_tf=True, stop_words="english", max_features=50000)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=2.0)),
    ])
    model.fit(train["text"], train["label"])

    val_pred = model.predict(val["text"])
    results["tfidf_logreg_val"] = scores(val["label"], val_pred)
    print("\nVALIDATION")
    print(classification_report(val["label"], val_pred))

    # Final, one-time evaluation on the untouched test set
    test_pred = model.predict(test["text"])
    results["tfidf_logreg_test"] = scores(test["label"], test_pred)
    print("TEST (final, report this number)")
    print(classification_report(test["label"], test_pred))

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    # Record which scikit-learn version trained the model; the backend warns if it later differs.
    meta = {
        "sklearn_version": sklearn.__version__,
        "trained_on": date.today().isoformat(),
        "n_train": len(train),
        "val_macro_f1": results["tfidf_logreg_val"]["macro_f1"],
        "test_macro_f1": results["tfidf_logreg_test"]["macro_f1"],
    }
    MODEL_OUT.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved model -> {MODEL_OUT}")
    print(f"\nPIN THIS in backend/requirements.txt (replace the plain 'scikit-learn' line):\n    scikit-learn=={sklearn.__version__}")


if __name__ == "__main__":
    main()
