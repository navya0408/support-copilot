"""Shared paths and the train/val/test split (identical in every script)."""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app.textclean import clean_text  # noqa: E402,F401  (re-exported)

DATA_RAW = ROOT / "data" / "cfpb_raw_sample.csv"
DATA_CLEAN = ROOT / "data" / "cfpb_clean.csv"
RESULTS_DIR = ROOT / "ml" / "results"
SEED = 42


def load_splits():
    """70% train / 15% validation / 15% test, stratified by label."""
    df = pd.read_csv(DATA_CLEAN)
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)
    return train, val, test
