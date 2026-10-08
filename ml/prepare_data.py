"""Clean the raw CFPB sample.

Input : data/cfpb_raw_sample.csv   (the 20,000-row sample you downloaded from Colab)
Output: data/cfpb_clean.csv

Run:  python ml/prepare_data.py
"""
import numpy as np
import pandas as pd

from common import DATA_CLEAN, DATA_RAW, clean_text

# 11 CFPB products -> 7 classes (tiny classes are merged into related ones)
LABEL_MAP = {
    "Credit reporting or other personal consumer reports": "CREDIT_REPORTING",
    "Debt collection": "DEBT_COLLECTION",
    "Debt or credit management": "DEBT_COLLECTION",
    "Checking or savings account": "BANK_ACCOUNT",
    "Credit card": "CREDIT_CARD",
    "Money transfer, virtual currency, or money service": "MONEY_TRANSFER_PREPAID",
    "Prepaid card": "MONEY_TRANSFER_PREPAID",
    "Mortgage": "MORTGAGE",
    "Vehicle loan or lease": "LOANS",
    "Payday loan, title loan, personal loan, or advance loan": "LOANS",
    "Student loan": "LOANS",
}


def main():
    df = pd.read_csv(DATA_RAW)
    print("raw rows:", len(df))

    df = df.replace("null", np.nan)                      # missing values were stored as text
    df = df.dropna(subset=["consumer_complaint_narrative", "product"])

    before = len(df)
    df = df.drop_duplicates(subset=["consumer_complaint_narrative"])
    print(f"dropped {before - len(df)} duplicate narratives")

    df["text"] = df["consumer_complaint_narrative"].apply(clean_text)
    df["n_words"] = df["text"].str.split().str.len()
    df = df[df["n_words"] >= 10]                         # drop near-empty complaints

    df["label"] = df["product"].map(LABEL_MAP)
    print("unmapped products:", df["label"].isna().sum())
    df = df.dropna(subset=["label"])

    print("\nclass distribution:\n", df["label"].value_counts())
    print("majority-class accuracy:", round(df["label"].value_counts(normalize=True).iloc[0], 4))

    DATA_CLEAN.parent.mkdir(exist_ok=True)
    df[["complaint_id", "text", "label", "tags", "n_words"]].to_csv(DATA_CLEAN, index=False)
    print(f"\nsaved {len(df)} rows -> {DATA_CLEAN}")


if __name__ == "__main__":
    main()
