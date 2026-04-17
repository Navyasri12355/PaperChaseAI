"""
Phase 3 — Transformer Fine-tuning
SciBERT (allenai/scibert_scivocab_uncased) for main_category prediction.

Usage:
    python train_scibert.py [--task main_category|sub_category] [--epochs 5] [--batch_size 32]

Outputs (all in outputs/scibert_{task}/):
    best_model/          — HuggingFace checkpoint (model + tokenizer)
    training_log.csv     — loss + val macro F1 per epoch
    test_report.txt      — full classification report on test set
    confusion_matrix.png — confusion matrix on test set
    training_curves.png  — loss + F1 curves
"""

import os
import time
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--task",       default="main_category",
                    choices=["main_category", "sub_category"])
parser.add_argument("--model_name", default="allenai/scibert_scivocab_uncased")
parser.add_argument("--max_len",    type=int, default=512)
parser.add_argument("--epochs",     type=int, default=5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr",         type=float, default=2e-5)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--patience",   type=int, default=2,
                    help="Early stopping patience (epochs without val F1 improvement)")
args = parser.parse_args()

TASK       = args.task
LABEL_COL  = "cat_label" if TASK == "main_category" else "sub_label"
LE_PATH    = f"outputs/le_{TASK}.pkl"
OUT_DIR    = f"outputs/scibert_{TASK}"
CKPT_DIR   = f"{OUT_DIR}/best_model"
os.makedirs(CKPT_DIR, exist_ok=True)

BASELINE_F1 = 0.7526 if TASK == "main_category" else 0.4702

print(f"\n{'='*60}")
print(f"  Task:      {TASK}")
print(f"  Model:     {args.model_name}")
print(f"  Max len:   {args.max_len}")
print(f"  Epochs:    {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
print(f"  Baseline to beat: {BASELINE_F1}")
print(f"{'='*60}\n")

# ── Device ─────────────────────────────────────────────────────────────────────
device = (
    torch.device("cuda")    if torch.cuda.is_available()  else
    torch.device("mps")     if torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Device: {device}")
if device.type == "cpu":
    print("  ⚠ No GPU detected. Training will be slow.")
    print("  Consider using Google Colab (free T4) or Kaggle (free P100).")
    print("  See README_COLAB.md for instructions.\n")

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data …")
train = pd.read_csv("outputs/train.csv")
val   = pd.read_csv("outputs/val.csv")
test  = pd.read_csv("outputs/test.csv")
le    = joblib.load(LE_PATH)
num_labels = len(le.classes_)
print(f"  Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print(f"  Num labels: {num_labels}")

# ── Class weights for imbalanced classes ───────────────────────────────────────
label_counts = np.bincount(train[LABEL_COL].values, minlength=num_labels).astype(float)
class_weights = torch.tensor(
    (label_counts.sum() / (num_labels * label_counts)).clip(max=10),
    dtype=torch.float
).to(device)
print(f"  Class weights computed (max={class_weights.max():.2f}, min={class_weights.min():.2f})")

# ── Tokenizer ──────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer: {args.model_name} …")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# ── Dataset ────────────────────────────────────────────────────────────────────
class PaperDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label_col):
        self.texts  = df["text"].fillna("").tolist()
        self.labels = df[label_col].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Use fewer workers on CPU to avoid overhead
num_workers = 4 if device.type != "cpu" else 0

train_ds = PaperDataset(train, tokenizer, args.max_len, LABEL_COL)
val_ds   = PaperDataset(val,   tokenizer, args.max_len, LABEL_COL)
test_ds  = PaperDataset(test,  tokenizer, args.max_len, LABEL_COL)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size * 2, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size * 2, shuffle=False,
                          num_workers=num_workers, pin_memory=(device.type == "cuda"))

# ── Model ──────────────────────────────────────────────────────────────────────
print(f"Loading model …")
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)
model.to(device)

# ── Optimizer + scheduler ──────────────────────────────────────────────────────
total_steps   = len(train_loader) * args.epochs
warmup_steps  = int(total_steps * args.warmup_ratio)

optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

loss_fn = nn.CrossEntropyLoss(weight=class_weights)

print(f"  Total steps: {total_steps:,}  |  Warmup steps: {warmup_steps:,}")

# ── Eval helper ───────────────────────────────────────────────────────────────
def evaluate(loader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            total_loss += loss.item()

            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, macro_f1, accuracy, np.array(all_preds), np.array(all_labels)

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nStarting training …\n")
log = []
best_val_f1  = 0.0
patience_ctr = 0

for epoch in range(1, args.epochs + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader, 1):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = loss_fn(outputs.logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()

        if step % 50 == 0 or step == len(train_loader):
            avg = epoch_loss / step
            elapsed = time.time() - t0
            print(f"  Epoch {epoch}/{args.epochs}  step {step}/{len(train_loader)}"
                  f"  loss={avg:.4f}  elapsed={elapsed:.0f}s", end="\r")

    train_loss = epoch_loss / len(train_loader)
    val_loss, val_f1, val_acc, _, _ = evaluate(val_loader)
    epoch_time = time.time() - t0

    print(f"\n  Epoch {epoch}  train_loss={train_loss:.4f}  "
          f"val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}  "
          f"val_acc={val_acc:.4f}  time={epoch_time:.0f}s")

    log.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "val_loss":   round(val_loss, 4),
        "val_macro_f1": round(val_f1, 4),
        "val_accuracy": round(val_acc, 4),
    })

    if val_f1 > best_val_f1:
        best_val_f1  = val_f1
        patience_ctr = 0
        model.save_pretrained(CKPT_DIR)
        tokenizer.save_pretrained(CKPT_DIR)
        print(f"  ✓ New best val macro F1: {best_val_f1:.4f} — checkpoint saved")
    else:
        patience_ctr += 1
        print(f"  No improvement ({patience_ctr}/{args.patience})")
        if patience_ctr >= args.patience:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

# ── Save training log ─────────────────────────────────────────────────────────
log_df = pd.DataFrame(log)
log_df.to_csv(f"{OUT_DIR}/training_log.csv", index=False)
print(f"\n✓ Training log → {OUT_DIR}/training_log.csv")

# ── Training curves ───────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"SciBERT — {TASK} Training Curves", fontweight="bold")

ax1.plot(log_df["epoch"], log_df["train_loss"], label="Train loss", marker="o")
ax1.plot(log_df["epoch"], log_df["val_loss"],   label="Val loss",   marker="o")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")

ax2.plot(log_df["epoch"], log_df["val_macro_f1"], label="Val macro F1", marker="o", color="green")
ax2.axhline(BASELINE_F1, color="red", linestyle="--", label=f"Baseline F1 ({BASELINE_F1})")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Macro F1"); ax2.legend(); ax2.set_title("Macro F1")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Training curves → {OUT_DIR}/training_curves.png")

# ── Test set evaluation ───────────────────────────────────────────────────────
print("\nLoading best checkpoint for test evaluation …")
model = AutoModelForSequenceClassification.from_pretrained(CKPT_DIR)
model.to(device)

_, test_f1, test_acc, test_preds, test_labels = evaluate(test_loader)
test_wf1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)

print(f"\n{'='*60}")
print(f"  TEST RESULTS — {TASK}")
print(f"  Accuracy:    {test_acc:.4f}")
print(f"  Macro F1:    {test_f1:.4f}  (baseline: {BASELINE_F1})")
print(f"  Weighted F1: {test_wf1:.4f}")
beat = "✓ BEATS baseline" if test_f1 > BASELINE_F1 else "✗ Did NOT beat baseline"
print(f"  {beat}")
print(f"{'='*60}\n")

# ── Per-class report ──────────────────────────────────────────────────────────
report = classification_report(
    test_labels, test_preds,
    target_names=le.classes_,
    zero_division=0
)

per_class_f1 = f1_score(test_labels, test_preds, average=None, zero_division=0)
class_f1_df  = pd.DataFrame({
    "class":   le.classes_,
    "f1":      per_class_f1,
    "support": np.bincount(test_labels, minlength=num_labels),
}).sort_values("f1", ascending=False)

report_lines = [
    f"SciBERT — {TASK} — Test Set Report",
    "=" * 60,
    f"Accuracy:    {test_acc:.4f}",
    f"Macro F1:    {test_f1:.4f}  (baseline to beat: {BASELINE_F1})",
    f"Weighted F1: {test_wf1:.4f}",
    beat,
    "",
    "Full Classification Report:",
    report,
    "",
    "Top 5 best predicted classes:",
    class_f1_df.head(5).to_string(index=False),
    "",
    "Top 5 worst predicted classes:",
    class_f1_df.tail(5).to_string(index=False),
    "",
    "Zero F1 classes (if any):",
    class_f1_df[class_f1_df["f1"] == 0].to_string(index=False) or "  None ✓",
]

report_path = f"{OUT_DIR}/test_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(report_lines))
print(f"✓ Test report → {report_path}")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm     = confusion_matrix(test_labels, test_preds)
labels = le.classes_
max_classes = 30

if len(labels) > max_classes:
    support = cm.sum(axis=1)
    top_idx = np.argsort(support)[-max_classes:][::-1]
    cm      = cm[np.ix_(top_idx, top_idx)]
    labels  = labels[top_idx]

fig_h = max(8, len(labels) * 0.38)
fig, ax = plt.subplots(figsize=(fig_h + 2, fig_h))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            ax=ax, linewidths=0.3, cbar=False, annot_kws={"size": 7})
ax.set_title(f"SciBERT — {TASK} Confusion Matrix (Test Set)", fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
cm_path = f"{OUT_DIR}/confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Confusion matrix → {cm_path}")

# ── Update master comparison table ───────────────────────────────────────────
master_csv = "outputs/baseline_results.csv"
if os.path.exists(master_csv):
    master = pd.read_csv(master_csv)
else:
    master = pd.DataFrame()

new_rows = pd.DataFrame([
    {
        "model": "SciBERT", "task": TASK, "split": "test",
        "accuracy": round(test_acc, 4),
        "macro_f1": round(test_f1, 4),
        "weighted_f1": round(test_wf1, 4),
        "train_time_s": round(sum(log_df.get("epoch", [0])) * 0, 0),  # placeholder
    }
])
master = pd.concat([master, new_rows], ignore_index=True)
master.to_csv(master_csv, index=False)
print(f"✓ Master results table updated → {master_csv}")

print(f"\n{'='*60}")
print(f"  PHASE 3 COMPLETE — {TASK}")
print(f"  Best val macro F1:  {best_val_f1:.4f}")
print(f"  Test macro F1:      {test_f1:.4f}")
print(f"  Checkpoint saved:   {CKPT_DIR}")
print(f"{'='*60}\n")