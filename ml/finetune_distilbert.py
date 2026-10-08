"""Fine-tune DistilBERT on the same split (run on a GPU, e.g. Google Colab T4).

Colab usage:
    1. Runtime > Change runtime type > T4 GPU
    2. Get this repo into Colab (git clone your GitHub repo) and put
       data/cfpb_clean.csv in the data/ folder, then:
           !python ml/finetune_distilbert.py

Output: ml/artifacts/distilbert_cfpb/, ml/results/distilbert_metrics.json
"""
import json
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from common import ROOT, RESULTS_DIR, load_splits

MODEL, MAX_LEN, BATCH, EPOCHS, LR = "distilbert-base-uncased", 256, 16, 3, 3e-5
OUT_DIR = ROOT / "ml" / "artifacts" / "distilbert_cfpb"
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

train, val, test = load_splits()
classes = sorted(train["label"].unique())
label2id = {c: i for i, c in enumerate(classes)}
id2label = {i: c for c, i in label2id.items()}
tok = AutoTokenizer.from_pretrained(MODEL)


class TicketDS(Dataset):
    def __init__(self, df):
        self.enc = tok(list(df["text"]), truncation=True, max_length=MAX_LEN)
        self.labels = [label2id[l] for l in df["label"]]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


collator = DataCollatorWithPadding(tok)
train_dl = DataLoader(TicketDS(train), batch_size=BATCH, shuffle=True, collate_fn=collator)
val_dl = DataLoader(TicketDS(val), batch_size=64, collate_fn=collator)
test_dl = DataLoader(TicketDS(test), batch_size=64, collate_fn=collator)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(classes), id2label=id2label, label2id=label2id).to(device)

y_train = np.array([label2id[l] for l in train["label"]])
weights = compute_class_weight("balanced", classes=np.arange(len(classes)), y=y_train)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(device))
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def evaluate(dl):
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            preds += logits.argmax(-1).cpu().tolist()
            gold += labels.cpu().tolist()
    return np.array(gold), np.array(preds)


best_f1, best_state = 0.0, None
for epoch in range(EPOCHS):
    model.train()
    t0, running = time.time(), 0.0
    for step, batch in enumerate(train_dl):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
            logits = model(**batch).logits
        loss = loss_fn(logits.float(), labels)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.item()
        if (step + 1) % 100 == 0:
            print(f"  epoch {epoch + 1} step {step + 1}/{len(train_dl)} loss={running / 100:.3f}")
            running = 0.0
    gold, preds = evaluate(val_dl)
    mf1 = f1_score(gold, preds, average="macro")
    print(f"Epoch {epoch + 1}: val accuracy={(gold == preds).mean():.3f} macro-F1={mf1:.3f} ({time.time() - t0:.0f}s)")
    if mf1 > best_f1:
        best_f1 = mf1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
vg, vp = evaluate(val_dl)
tg, tp = evaluate(test_dl)
results = {
    "distilbert_val": {"accuracy": round(float((vg == vp).mean()), 4), "macro_f1": round(float(f1_score(vg, vp, average="macro")), 4)},
    "distilbert_test": {"accuracy": round(float((tg == tp).mean()), 4), "macro_f1": round(float(f1_score(tg, tp, average="macro")), 4)},
}
print(classification_report(vg, vp, target_names=classes))
print(json.dumps(results, indent=2))

OUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT_DIR)
tok.save_pretrained(OUT_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "distilbert_metrics.json").write_text(json.dumps(results, indent=2))
