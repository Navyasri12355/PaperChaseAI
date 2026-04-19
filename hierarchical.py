"""
Phase 4 — Hierarchical Classification (Constrained Sequential)
===============================================================
Strategy: predict main_category first, then route to a per-category
          sub-classifier. Invalid subcategories are masked out by
          construction — the model physically cannot predict a Physics
          subcategory for a Mathematics paper.

Why this beats flat sub_category classification:
  - Each sub-problem: ~10–20 classes instead of 117
  - Confused cross-category codes (ML, CO, GR etc.) become impossible
  - More training signal per class within each sub-classifier
  - val→test gap shrinks (less class imbalance per sub-problem)

Usage:
    # Step 1 — train all per-category sub-classifiers (TF-IDF + LR, fast)
    python hierarchical.py --mode train

    # Step 2 — evaluate joint accuracy on test set
    python hierarchical.py --mode eval

    # Step 3 — interactive single paper prediction
    python hierarchical.py --mode predict

Outputs → outputs/hierarchical/
"""

import os, argparse, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
OUT_DIR    = "outputs/hierarchical"
MODEL_DIR  = f"{OUT_DIR}/models"
os.makedirs(MODEL_DIR, exist_ok=True)

TFIDF_KWARGS = dict(
    max_features=100_000,
    sublinear_tf=True,
    ngram_range=(1, 2),
    min_df=2,
    strip_accents="unicode",
)

# ── Load ───────────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv("outputs/train.csv")
    val   = pd.read_csv("outputs/val.csv")
    test  = pd.read_csv("outputs/test.csv")
    le_cat = joblib.load("outputs/le_main_category.pkl")
    le_sub = joblib.load("outputs/le_sub_category.pkl")
    return train, val, test, le_cat, le_sub

# ── Build category → subcategory mapping from training data ───────────────────
def build_hierarchy(train, le_cat, le_sub):
    """
    Returns:
        hierarchy: dict  {category_name: [sub_label_int, ...]}
        cat_to_subs: dict {category_name: [sub_name, ...]}
    """
    hierarchy    = {}
    cat_to_subs  = {}
    for cat_name in le_cat.classes_:
        cat_mask  = train["main_category"] == cat_name
        sub_names = train.loc[cat_mask, "sub_category"].unique().tolist()
        sub_ints  = sorted(train.loc[cat_mask, "sub_label"].unique().tolist())
        hierarchy[cat_name]   = sub_ints
        cat_to_subs[cat_name] = sub_names
    return hierarchy, cat_to_subs

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN MODE
# ══════════════════════════════════════════════════════════════════════════════
def train_mode():
    train, val, test, le_cat, le_sub = load_data()
    hierarchy, cat_to_subs = build_hierarchy(train, le_cat, le_sub)

    print("\n=== Training per-category sub-classifiers ===\n")
    print(f"{'Category':<35} {'Sub classes':>11} {'Train samples':>14} {'Val macro F1':>13}")
    print("-" * 76)

    cat_results = []

    for cat_name in sorted(le_cat.classes_):
        # ── Filter to this category ─────────────────────────────────────────
        tr  = train[train["main_category"] == cat_name].reset_index(drop=True)
        v   = val[val["main_category"]     == cat_name].reset_index(drop=True)
        sub_classes = sorted(tr["sub_category"].unique())
        n_sub = len(sub_classes)

        if n_sub == 1:
            # Only one subcategory — trivial classifier, save a sentinel
            print(f"  {cat_name:<33} {n_sub:>11} {len(tr):>14}  (trivial — 1 class)")
            joblib.dump({"trivial": sub_classes[0]},
                        f"{MODEL_DIR}/sub_{cat_name.replace(' ','_')}.pkl")
            cat_results.append(dict(category=cat_name, n_sub=n_sub,
                                    train_n=len(tr), val_macro_f1=1.0))
            continue

        # ── Local label encoding (0..n_sub-1 within this category) ─────────
        local_le = {s: i for i, s in enumerate(sub_classes)}
        y_tr = tr["sub_category"].map(local_le).values
        y_v  = v["sub_category"].map(local_le).values

        # ── Choose model: SVC for large, LR for small ───────────────────────
        if len(tr) > 500:
            clf = Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf",   CalibratedClassifierCV(
                    LinearSVC(max_iter=2000, C=1.0, class_weight="balanced"),
                    cv=3
                )),
            ])
        else:
            clf = Pipeline([
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf",   LogisticRegression(max_iter=1000, C=5.0,
                                             class_weight="balanced",
                                             solver="lbfgs", n_jobs=-1)),
            ])

        clf.fit(tr["text"], y_tr)

        # ── Val evaluation ──────────────────────────────────────────────────
        if len(v) > 0 and len(np.unique(y_v)) > 1:
            val_preds  = clf.predict(v["text"])
            val_mf1    = f1_score(y_v, val_preds, average="macro", zero_division=0)
        else:
            val_mf1 = float("nan")

        print(f"  {cat_name:<33} {n_sub:>11} {len(tr):>14} {val_mf1:>13.4f}")

        # ── Save: model + local label mapping ───────────────────────────────
        joblib.dump(
            {"model": clf, "local_le": local_le,
             "sub_classes": sub_classes, "category": cat_name},
            f"{MODEL_DIR}/sub_{cat_name.replace(' ','_')}.pkl"
        )
        cat_results.append(dict(category=cat_name, n_sub=n_sub,
                                train_n=len(tr), val_macro_f1=round(val_mf1, 4)))

    # ── Train top-level category classifier ──────────────────────────────────
    print("\n=== Training top-level category classifier ===")
    cat_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
        ("clf",   LogisticRegression(max_iter=1000, C=5.0,
                                     class_weight="balanced",
                                     solver="lbfgs", n_jobs=-1)),
    ])
    cat_pipe.fit(train["text"], train["cat_label"])
    val_cat_preds = cat_pipe.predict(val["text"])
    cat_val_f1 = f1_score(val["cat_label"], val_cat_preds,
                          average="macro", zero_division=0)
    print(f"  Category classifier val macro F1: {cat_val_f1:.4f}")
    joblib.dump(cat_pipe, f"{MODEL_DIR}/cat_classifier.pkl")

    # ── Save hierarchy map ────────────────────────────────────────────────────
    joblib.dump({"hierarchy": hierarchy, "cat_to_subs": cat_to_subs},
                f"{MODEL_DIR}/hierarchy.pkl")

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(cat_results)
    results_df.to_csv(f"{OUT_DIR}/per_category_val_results.csv", index=False)
    print(f"\n✓ Per-category results → {OUT_DIR}/per_category_val_results.csv")
    print(f"✓ Models saved → {MODEL_DIR}/")
    print("\nRun with --mode eval to evaluate on the test set.")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT HELPER (used by both eval and predict modes)
# ══════════════════════════════════════════════════════════════════════════════
def load_pipeline(le_cat, le_sub):
    cat_pipe  = joblib.load(f"{MODEL_DIR}/cat_classifier.pkl")
    hier_data = joblib.load(f"{MODEL_DIR}/hierarchy.pkl")
    sub_models = {}
    for cat_name in le_cat.classes_:
        path = f"{MODEL_DIR}/sub_{cat_name.replace(' ','_')}.pkl"
        if os.path.exists(path):
            sub_models[cat_name] = joblib.load(path)
    return cat_pipe, sub_models, hier_data

def predict_batch(texts, cat_pipe, sub_models, le_cat, le_sub, top_k=3):
    """
    Returns list of dicts with:
        predicted_category, predicted_sub_category,
        category_confidence, sub_confidence,
        top_k_categories (list of (cat, prob))
    """
    cat_probs  = cat_pipe.predict_proba(texts)   # (N, n_cat)
    cat_preds  = cat_probs.argmax(axis=1)
    cat_names  = le_cat.inverse_transform(cat_preds)

    results = []
    for i, (text, cat_name, cat_prob_vec) in enumerate(
            zip(texts, cat_names, cat_probs)):

        cat_conf = cat_prob_vec.max()
        top_cats = sorted(
            zip(le_cat.classes_, cat_prob_vec),
            key=lambda x: -x[1]
        )[:top_k]

        sub_data = sub_models.get(cat_name, {})

        if "trivial" in sub_data:
            sub_name = sub_data["trivial"]
            sub_conf = 1.0
        elif "model" in sub_data:
            local_le    = sub_data["local_le"]
            sub_classes = sub_data["sub_classes"]
            sub_pipe    = sub_data["model"]

            sub_probs = sub_pipe.predict_proba([text])[0]
            sub_pred  = sub_probs.argmax()
            sub_name  = sub_classes[sub_pred]
            sub_conf  = sub_probs.max()
        else:
            sub_name = "unknown"
            sub_conf = 0.0

        results.append({
            "predicted_category":     cat_name,
            "predicted_sub_category": sub_name,
            "category_confidence":    round(float(cat_conf), 4),
            "sub_confidence":         round(float(sub_conf), 4),
            "top_k_categories":       [(c, round(float(p), 4)) for c, p in top_cats],
        })
    return results

# ══════════════════════════════════════════════════════════════════════════════
# EVAL MODE
# ══════════════════════════════════════════════════════════════════════════════
def eval_mode():
    train, val, test, le_cat, le_sub = load_data()
    cat_pipe, sub_models, _ = load_pipeline(le_cat, le_sub)

    BASELINE_SUB_F1  = 0.4702
    BASELINE_CAT_F1  = 0.7526
    SCIBERT_CAT_F1   = None   # fill in if you have it

    print("\n=== Hierarchical evaluation on TEST set ===\n")

    preds = predict_batch(
        test["text"].tolist(), cat_pipe, sub_models, le_cat, le_sub
    )
    pred_cats = [p["predicted_category"]     for p in preds]
    pred_subs = [p["predicted_sub_category"] for p in preds]
    true_cats = test["main_category"].tolist()
    true_subs = test["sub_category"].tolist()

    # ── Category metrics ──────────────────────────────────────────────────────
    cat_acc  = accuracy_score(true_cats, pred_cats)
    cat_mf1  = f1_score(true_cats, pred_cats, average="macro",     zero_division=0)
    cat_wf1  = f1_score(true_cats, pred_cats, average="weighted",  zero_division=0)

    # ── Sub metrics (flat — compare against baseline) ─────────────────────────
    sub_acc  = accuracy_score(true_subs, pred_subs)
    sub_mf1  = f1_score(true_subs, pred_subs, average="macro",    zero_division=0)
    sub_wf1  = f1_score(true_subs, pred_subs, average="weighted", zero_division=0)

    # ── Joint accuracy ────────────────────────────────────────────────────────
    joint_correct = sum(c == tc and s == ts
                        for c, s, tc, ts in
                        zip(pred_cats, pred_subs, true_cats, true_subs))
    joint_acc = joint_correct / len(test)

    # ── Hierarchical consistency ──────────────────────────────────────────────
    # Was the predicted sub actually valid for the predicted category?
    hier_data   = joblib.load(f"{MODEL_DIR}/hierarchy.pkl")
    cat_to_subs = hier_data["cat_to_subs"]
    consistent  = sum(
        ps in cat_to_subs.get(pc, [])
        for pc, ps in zip(pred_cats, pred_subs)
    )
    hier_consistency = consistent / len(test)

    print(f"{'─'*55}")
    print(f"  CATEGORY")
    print(f"    Accuracy:       {cat_acc:.4f}")
    print(f"    Macro F1:       {cat_mf1:.4f}  (baseline: {BASELINE_CAT_F1})")
    print(f"    Weighted F1:    {cat_wf1:.4f}")
    print(f"{'─'*55}")
    print(f"  SUB-CATEGORY (flat macro F1 against baseline)")
    print(f"    Accuracy:       {sub_acc:.4f}")
    print(f"    Macro F1:       {sub_mf1:.4f}  (baseline: {BASELINE_SUB_F1})")
    print(f"    Weighted F1:    {sub_wf1:.4f}")
    beat = "✓ BEATS baseline" if sub_mf1 > BASELINE_SUB_F1 else "✗ Below baseline"
    print(f"    {beat}")
    print(f"{'─'*55}")
    print(f"  JOINT")
    print(f"    Joint accuracy (both correct): {joint_acc:.4f}")
    print(f"    Hierarchical consistency:      {hier_consistency:.4f}")
    print(f"{'─'*55}\n")

    # ── Per-category sub F1 breakdown ─────────────────────────────────────────
    print("  Per-category sub-classifier F1 on test:\n")
    cat_sub_f1 = []
    for cat_name in sorted(le_cat.classes_):
        mask  = [tc == cat_name for tc in true_cats]
        t_sub = [ts for ts, m in zip(true_subs, mask) if m]
        p_sub = [ps for ps, m in zip(pred_subs, mask) if m]
        if len(t_sub) == 0:
            continue
        mf1 = f1_score(t_sub, p_sub, average="macro", zero_division=0)
        n   = len(t_sub)
        n_sub_classes = len(set(t_sub))
        print(f"    {cat_name:<35} n={n:>4}  sub_classes={n_sub_classes:>3}  macro_f1={mf1:.4f}")
        cat_sub_f1.append(dict(category=cat_name, n=n,
                               n_sub_classes=n_sub_classes, test_macro_f1=round(mf1, 4)))

    # ── Save full report ──────────────────────────────────────────────────────
    report_lines = [
        "Hierarchical Classifier — Test Report",
        "="*60,
        f"Category  macro F1:         {cat_mf1:.4f}  (baseline: {BASELINE_CAT_F1})",
        f"Sub-cat   macro F1 (flat):   {sub_mf1:.4f}  (baseline: {BASELINE_SUB_F1})  {beat}",
        f"Joint accuracy:              {joint_acc:.4f}",
        f"Hierarchical consistency:    {hier_consistency:.4f}",
        "",
        "--- Category classification report ---",
        classification_report(true_cats, pred_cats, zero_division=0),
        "",
        "--- Sub-category classification report ---",
        classification_report(true_subs, pred_subs, zero_division=0),
    ]
    with open(f"{OUT_DIR}/test_report.txt", "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n✓ Full report → {OUT_DIR}/test_report.txt")

    # ── Update master results table ───────────────────────────────────────────
    master_csv = "outputs/baseline_results.csv"
    master = pd.read_csv(master_csv) if os.path.exists(master_csv) else pd.DataFrame()
    new_rows = pd.DataFrame([
        {"model": "Hierarchical-LR", "task": "sub_category",  "split": "test",
         "accuracy": round(sub_acc,4), "macro_f1": round(sub_mf1,4),
         "weighted_f1": round(sub_wf1,4), "train_time_s": 0},
        {"model": "Hierarchical-LR", "task": "main_category", "split": "test",
         "accuracy": round(cat_acc,4), "macro_f1": round(cat_mf1,4),
         "weighted_f1": round(cat_wf1,4), "train_time_s": 0},
        {"model": "Hierarchical-LR", "task": "joint",         "split": "test",
         "accuracy": round(joint_acc,4), "macro_f1": round(joint_acc,4),
         "weighted_f1": round(hier_consistency,4), "train_time_s": 0},
    ])
    pd.concat([master, new_rows], ignore_index=True).to_csv(master_csv, index=False)
    print(f"✓ Master results table updated → {master_csv}")

    # ── Confusion matrix — categories ─────────────────────────────────────────
    cats = sorted(le_cat.classes_)
    cm   = confusion_matrix(true_cats, pred_cats, labels=cats)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cats, yticklabels=cats,
                ax=ax, linewidths=0.5, annot_kws={"size": 9})
    ax.set_title("Hierarchical — Category Confusion Matrix (Test)", fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_category.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Category confusion matrix → {OUT_DIR}/confusion_category.png")

    # ── Per-category sub F1 bar chart ─────────────────────────────────────────
    df_cat_f1 = pd.DataFrame(cat_sub_f1).sort_values("test_macro_f1", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#C44E52" if v < BASELINE_SUB_F1 else "#55A868"
              for v in df_cat_f1["test_macro_f1"]]
    ax.barh(df_cat_f1["category"], df_cat_f1["test_macro_f1"],
            color=colors, alpha=0.85)
    ax.axvline(BASELINE_SUB_F1, color="red",  linestyle="--", lw=1.5,
               label=f"Flat baseline ({BASELINE_SUB_F1})")
    ax.set_xlabel("Sub-category macro F1")
    ax.set_title("Per-category Sub-classifier F1 (Test Set)\nGreen = beats flat baseline",
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/per_category_sub_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Per-category sub F1 chart → {OUT_DIR}/per_category_sub_f1.png")

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT MODE — single paper
# ══════════════════════════════════════════════════════════════════════════════
def predict_mode():
    _, _, _, le_cat, le_sub = load_data()
    cat_pipe, sub_models, _ = load_pipeline(le_cat, le_sub)

    print("\n=== Single paper prediction ===")
    print("Paste title and abstract. Empty line to submit. Ctrl+C to quit.\n")

    while True:
        try:
            print("Title: ", end=""); title = input().strip()
            print("Abstract (press Enter twice when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            abstract = " ".join(lines).strip()

            if not title and not abstract:
                print("Nothing entered.\n"); continue

            text = f"{title.lower()} [SEP] {abstract.lower()}"
            result = predict_batch([text], cat_pipe, sub_models, le_cat, le_sub)[0]

            print(f"\n  ┌─ Prediction ─────────────────────────────────────┐")
            print(f"  │ Category:     {result['predicted_category']:<35}│")
            print(f"  │   confidence: {result['category_confidence']:.4f}                              │")
            print(f"  │ Sub-category: {result['predicted_sub_category']:<35}│")
            print(f"  │   confidence: {result['sub_confidence']:.4f}                              │")
            print(f"  ├─ Top-3 categories ───────────────────────────────┤")
            for cat, prob in result["top_k_categories"]:
                print(f"  │   {cat:<35} {prob:.4f}         │")
            print(f"  └──────────────────────────────────────────────────┘\n")

        except KeyboardInterrupt:
            print("\nBye.")
            break

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train",
                        choices=["train", "eval", "predict"])
    args = parser.parse_args()

    if args.mode == "train":
        train_mode()
    elif args.mode == "eval":
        eval_mode()
    elif args.mode == "predict":
        predict_mode()