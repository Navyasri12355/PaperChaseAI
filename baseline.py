"""
Phase 2 — Baseline Model
TF-IDF + Logistic Regression and LinearSVC
for both main_category and sub_category prediction.

Usage:
    python baseline.py

Outputs (all in outputs/):
    baseline_results.csv       — comparison table
    baseline_confusion_cat.png — confusion matrix for best category model
    baseline_confusion_sub.png — confusion matrix for best sub_category model
    baseline_report.txt        — full per-class F1 report
"""

import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
TRAIN_CSV  = f"{OUTPUT_DIR}/train.csv"
VAL_CSV    = f"{OUTPUT_DIR}/val.csv"
TEST_CSV   = f"{OUTPUT_DIR}/test.csv"
LE_CAT     = f"{OUTPUT_DIR}/le_main_category.pkl"
LE_SUB     = f"{OUTPUT_DIR}/le_sub_category.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading splits …")
train = pd.read_csv(TRAIN_CSV)
val   = pd.read_csv(VAL_CSV)
test  = pd.read_csv(TEST_CSV)

le_cat = joblib.load(LE_CAT)
le_sub = joblib.load(LE_SUB)

X_train, X_val, X_test = train["text"], val["text"], test["text"]

# ── Build pipelines ────────────────────────────────────────────────────────────
TFIDF_KWARGS = dict(
    max_features=100_000,
    sublinear_tf=True,
    ngram_range=(1, 2),
    min_df=2,
    strip_accents="unicode",
)

pipelines = {
    "LR":  Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf",   LogisticRegression(max_iter=1000, C=5.0,
                                     class_weight="balanced",
                                     solver="lbfgs",
                                     n_jobs=-1)),
    ]),
    "SVC": Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf",   LinearSVC(max_iter=2000, C=1.0,
                            class_weight="balanced")),
    ]),
}

# ── Helper ─────────────────────────────────────────────────────────────────────
results = []

def evaluate(name, task, model, X_tr, y_tr, X_val, y_val, X_te, y_te, le):
    print(f"\n  [{name} | {task}] Training …")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    for split_name, X_s, y_s in [("val", X_val, y_val), ("test", X_te, y_te)]:
        preds = model.predict(X_s)
        acc   = accuracy_score(y_s, preds)
        mf1   = f1_score(y_s, preds, average="macro", zero_division=0)
        wf1   = f1_score(y_s, preds, average="weighted", zero_division=0)
        print(f"    {split_name:5s} | acc={acc:.4f}  macro-F1={mf1:.4f}  weighted-F1={wf1:.4f}  "
              f"time={train_time:.1f}s")
        results.append({
            "model": name, "task": task, "split": split_name,
            "accuracy": round(acc, 4),
            "macro_f1": round(mf1, 4),
            "weighted_f1": round(wf1, 4),
            "train_time_s": round(train_time, 1),
        })

    return model   # return fitted model

# ── Run all four combinations ──────────────────────────────────────────────────
fitted = {}
for task, label_col in [("main_category", "cat_label"),
                         ("sub_category",  "sub_label")]:
    y_tr  = train[label_col].values
    y_val = val[label_col].values
    y_te  = test[label_col].values
    le    = le_cat if task == "main_category" else le_sub

    for model_name, pipe in pipelines.items():
        import copy
        pipe_copy = copy.deepcopy(pipe)
        fitted[(model_name, task)] = evaluate(
            model_name, task, pipe_copy,
            X_train, y_tr, X_val, y_val, X_test, y_te, le
        )

# ── Save comparison table ──────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_csv = f"{OUTPUT_DIR}/baseline_results.csv"
results_df.to_csv(results_csv, index=False)
print(f"\n✓ Results table → {results_csv}")
print(results_df.to_string(index=False))

# ── Per-class F1 report (test set) ────────────────────────────────────────────
report_lines = []

for task, label_col in [("main_category", "cat_label"),
                         ("sub_category",  "sub_label")]:
    le = le_cat if task == "main_category" else le_sub
    y_te = test[label_col].values

    for model_name in ["LR", "SVC"]:
        model = fitted[(model_name, task)]
        preds = model.predict(X_test)

        report = classification_report(
            y_te, preds,
            target_names=le.classes_,
            zero_division=0
        )
        header = f"\n{'='*70}\n{model_name} | {task} — Test Set Classification Report\n{'='*70}"
        report_lines.append(header)
        report_lines.append(report)

        # Top 5 best / worst per-class F1
        per_class_f1 = f1_score(y_te, preds, average=None, zero_division=0)
        class_f1_df  = pd.DataFrame({
            "class": le.classes_,
            "f1":    per_class_f1,
            "support": np.bincount(y_te, minlength=len(le.classes_))
        }).sort_values("f1", ascending=False)

        report_lines.append(f"\n--- Top 5 best predicted classes ---")
        report_lines.append(class_f1_df.head(5).to_string(index=False))
        report_lines.append(f"\n--- Top 5 worst predicted classes ---")
        report_lines.append(class_f1_df.tail(5).to_string(index=False))

report_txt = "\n".join(report_lines)
report_path = f"{OUTPUT_DIR}/baseline_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_txt)
print(f"✓ Per-class report → {report_path}")

# ── Confusion matrices (best model per task on test set) ──────────────────────
def plot_confusion(model, X_te, y_te, le, title, save_path, max_classes=30):
    preds = model.predict(X_te)
    cm    = confusion_matrix(y_te, preds)
    labels = le.classes_

    # If too many classes, show only top-N by support
    if len(labels) > max_classes:
        support = cm.sum(axis=1)
        top_idx = np.argsort(support)[-max_classes:][::-1]
        cm     = cm[np.ix_(top_idx, top_idx)]
        labels = labels[top_idx]
        title  = f"{title}\n(top {max_classes} classes by support)"

    fig_h = max(8, len(labels) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_h + 2, fig_h))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                ax=ax, linewidths=0.3, cbar=False,
                annot_kws={"size": 7})
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Confusion matrix → {save_path}")

# Pick best model per task (by test macro F1)
test_results = results_df[results_df["split"] == "test"]
for task, label_col, out_png in [
    ("main_category", "cat_label", f"{OUTPUT_DIR}/baseline_confusion_cat.png"),
    ("sub_category",  "sub_label", f"{OUTPUT_DIR}/baseline_confusion_sub.png"),
]:
    best_row   = test_results[test_results["task"] == task].sort_values("macro_f1", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = fitted[(best_model_name, task)]
    le = le_cat if task == "main_category" else le_sub
    y_te = test[label_col].values
    plot_confusion(
        best_model, X_test, y_te, le,
        title=f"{best_model_name} — {task} Confusion Matrix (Test Set)",
        save_path=out_png
    )

# ── Most confused pairs (for error analysis prep) ─────────────────────────────
confused_lines = ["\n\nMost Confused Category Pairs (main_category, test set)\n" + "="*60]
for model_name in ["LR", "SVC"]:
    model = fitted[(model_name, "main_category")]
    preds = model.predict(X_test)
    y_te  = test["cat_label"].values
    cm    = confusion_matrix(y_te, preds)
    np.fill_diagonal(cm, 0)
    flat   = cm.flatten()
    top10  = np.argsort(flat)[-10:][::-1]
    pairs  = [(divmod(i, len(le_cat.classes_))) for i in top10]
    confused_lines.append(f"\n{model_name}:")
    for true_i, pred_i in pairs:
        confused_lines.append(
            f"  {le_cat.classes_[true_i]:<35} → {le_cat.classes_[pred_i]:<35}  ({cm[true_i, pred_i]} errors)"
        )

with open(report_path, "a", encoding="utf-8") as f:
    f.write("\n".join(confused_lines))

print("\n" + "="*60)
print("PHASE 2 COMPLETE")
print("="*60)
print(f"  outputs/baseline_results.csv")
print(f"  outputs/baseline_report.txt")
print(f"  outputs/baseline_confusion_cat.png")
print(f"  outputs/baseline_confusion_sub.png")
print("\nBaseline macro F1 targets to beat in Phase 3:")
for _, row in test_results.sort_values(["task", "macro_f1"], ascending=[True, False]).drop_duplicates("task").iterrows():
    print(f"  {row['task']:20s}: {row['macro_f1']:.4f}  ({row['model']})")
