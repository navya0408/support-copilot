"""Ticket category classifier (TF-IDF + Logistic Regression trained by ml/train_baseline.py)."""
import json
import logging

import joblib
import numpy as np

from . import messages
from .config import MODEL_PATH
from .textclean import clean_text

log = logging.getLogger("support-copilot")
_model = None


def _check_version():
    """Warn if scikit-learn now differs from the version that trained the model."""
    meta_path = MODEL_PATH.with_suffix(".meta.json")
    if not meta_path.exists():
        log.info("No %s found (re-run ml/train_baseline.py to create it)", meta_path.name)
        return
    try:
        import sklearn
        trained_with = json.loads(meta_path.read_text()).get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            log.warning(
                "Model was trained with scikit-learn %s but %s is installed. "
                "Pin scikit-learn==%s in requirements.txt or retrain.",
                trained_with, sklearn.__version__, trained_with,
            )
    except Exception as exc:  # never block startup because of metadata
        log.warning("Could not read model metadata: %s", exc)


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(messages.CLASSIFIER_MISSING.format(path=MODEL_PATH))
        _check_version()
        _model = joblib.load(MODEL_PATH)
    return _model


def is_loaded() -> bool:
    return _model is not None


def predict(text: str):
    """Return (best_label, best_probability, top3_list)."""
    model = get_model()
    proba = model.predict_proba([clean_text(text)])[0]
    order = np.argsort(proba)[::-1]
    classes = model.classes_
    top = [{"label": str(classes[i]), "probability": round(float(proba[i]), 3)} for i in order[:3]]
    return top[0]["label"], top[0]["probability"], top
