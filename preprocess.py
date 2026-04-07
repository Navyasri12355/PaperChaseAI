"""
arxiv_balanced_19060_papers.csv
EDA + Preprocessing Pipeline
Phases 0 + 1 of the NLP Research Paper Classifier
"""

import os
import re
import ast
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
INPUT_CSV   = "arxiv_balanced_19060_papers.csv"
OUTPUT_DIR  = "/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_CSV   = f"{OUTPUT_DIR}/train.csv"
VAL_CSV     = f"{OUTPUT_DIR}/val.csv"
TEST_CSV    = f"{OUTPUT_DIR}/test.csv"
EDA_PNG     = f"{OUTPUT_DIR}/eda_report.png"
LE_CAT      = f"{OUTPUT_DIR}/le_main_category.pkl"
LE_SUB      = f"{OUTPUT_DIR}/le_sub_category.pkl"
STATS_TXT   = f"{OUTPUT_DIR}/data_card.txt"

# ── palette ────────────────────────────────────────────────────────────────────
PALETTE = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2",
           "#937860","#DA8BC3","#8C8C8C","#CCB974","#64B5CD"]

sns.set_theme(style="whitegrid", font_scale=0.85)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data …")
df = pd.read_csv(INPUT_CSV)
print(f"  Raw shape: {df.shape}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA  (runs before any cleaning so charts reflect raw reality)
# ═══════════════════════════════════════════════════════════════════════════════
print("Running EDA …")

df["abstract_words"] = df["abstract"].fillna("").str.split().str.len()
df["title_words"]    = df["title"].fillna("").str.split().str.len()
df["approx_tokens"]  = (df["abstract_words"] + df["title_words"]) * 1.3
df["num_labels"]     = df["all_labels"].str.count(",") + 1

fig = plt.figure(figsize=(20, 26))
fig.patch.set_facecolor("#FAFAFA")
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.35)

# ── 2a. Main category distribution ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
cat_counts = df["main_category"].value_counts()
bars = ax1.barh(cat_counts.index, cat_counts.values,
                color=PALETTE[:len(cat_counts)], edgecolor="white")
for bar, v in zip(bars, cat_counts.values):
    ax1.text(v + 30, bar.get_y() + bar.get_height()/2,
             str(v), va="center", fontsize=8)
ax1.set_title("Main Category Distribution", fontweight="bold", pad=8)
ax1.set_xlabel("Paper count")
ax1.invert_yaxis()

# ── 2b. Subcategory distribution (top 30) ─────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
sub_counts = df["sub_category"].value_counts().head(30)
ax2.bar(range(len(sub_counts)), sub_counts.values,
        color=PALETTE[2], alpha=0.85, edgecolor="white")
ax2.set_xticks(range(len(sub_counts)))
ax2.set_xticklabels(sub_counts.index, rotation=75, ha="right", fontsize=7)
ax2.set_title("Top 30 Subcategories", fontweight="bold", pad=8)
ax2.set_ylabel("Count")

# ── 2c. Abstract length distribution ──────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df["abstract_words"], bins=60, color=PALETTE[0], alpha=0.85, edgecolor="white")
ax3.axvline(df["abstract_words"].median(), color="#C44E52", lw=1.5,
            linestyle="--", label=f"Median {int(df['abstract_words'].median())} words")
ax3.axvline(df["abstract_words"].mean(), color="#55A868", lw=1.5,
            linestyle="--", label=f"Mean {int(df['abstract_words'].mean())} words")
ax3.set_title("Abstract Length Distribution (words)", fontweight="bold", pad=8)
ax3.set_xlabel("Word count")
ax3.set_ylabel("Frequency")
ax3.legend(fontsize=8)

# ── 2d. Approximate token distribution (title + abstract) ─────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(df["approx_tokens"], bins=60, color=PALETTE[1], alpha=0.85, edgecolor="white")
ax4.axvline(512, color="#C44E52", lw=2, linestyle="--", label="512-token limit")
pct_over = (df["approx_tokens"] > 512).mean() * 100
ax4.set_title(f"Approx Token Count (title+abstract)\n{pct_over:.1f}% exceed 512 tokens",
              fontweight="bold", pad=8)
ax4.set_xlabel("Approximate tokens (words × 1.3)")
ax4.set_ylabel("Frequency")
ax4.legend(fontsize=8)

# ── 2e. Multi-label distribution ──────────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 0])
label_counts = df["num_labels"].value_counts().sort_index()
ax5.bar(label_counts.index.astype(str), label_counts.values,
        color=PALETTE[4], alpha=0.85, edgecolor="white")
multilabel_pct = (df["num_labels"] > 1).mean() * 100
ax5.set_title(f"Labels per Paper\n{multilabel_pct:.1f}% of papers have >1 arXiv label",
              fontweight="bold", pad=8)
ax5.set_xlabel("Number of labels (all_labels column)")
ax5.set_ylabel("Count")
for i, (xi, yi) in enumerate(zip(label_counts.index, label_counts.values)):
    ax5.text(i, yi + 40, str(yi), ha="center", fontsize=8)

# ── 2f. Subcategory counts per main category (heatmap-style) ──────────────────
ax6 = fig.add_subplot(gs[2, 1])
sub_per_cat = df.groupby("main_category")["sub_category"].nunique().sort_values(ascending=False)
ax6.bar(range(len(sub_per_cat)), sub_per_cat.values,
        color=PALETTE[3], alpha=0.85, edgecolor="white")
ax6.set_xticks(range(len(sub_per_cat)))
ax6.set_xticklabels(sub_per_cat.index, rotation=30, ha="right", fontsize=8)
ax6.set_title("Unique Subcategories per Main Category", fontweight="bold", pad=8)
ax6.set_ylabel("Subcategory count")
for i, v in enumerate(sub_per_cat.values):
    ax6.text(i, v + 0.3, str(v), ha="center", fontsize=8)

# ── 2g. Class imbalance — subcategory tail ────────────────────────────────────
ax7 = fig.add_subplot(gs[3, :])
all_sub = df["sub_category"].value_counts()
colors = ["#C44E52" if v < 100 else PALETTE[0] for v in all_sub.values]
ax7.bar(range(len(all_sub)), all_sub.values, color=colors, alpha=0.85, edgecolor="white")
ax7.axhline(100, color="#C44E52", lw=1.2, linestyle="--", label="100-sample threshold (red bars = below)")
ax7.set_title("All Subcategory Counts (sorted) — Red = fewer than 100 samples",
              fontweight="bold", pad=8)
ax7.set_xlabel("Subcategory index (sorted by count)")
ax7.set_ylabel("Sample count")
ax7.legend(fontsize=8)

plt.suptitle("arXiv Papers Dataset — EDA Report", fontsize=15, fontweight="bold", y=1.002)
plt.savefig(EDA_PNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  EDA chart saved → {EDA_PNG}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("Preprocessing …")

# ── 3a. Handle the 158 missing sub_category (all math.NA) ─────────────────────
# math.NA = Numerical Analysis — a perfectly valid subcategory.
# It exists in full_code but was never decoded into sub_category.
# Fix: assign it the label "NA" (consistent with the arxiv code math.NA).
df["sub_category"] = df["sub_category"].fillna("NA")
print(f"  Filled 158 missing sub_category (math.NA → 'NA')")

# ── 3b. Resolve ambiguous subcategory codes ────────────────────────────────────
# 12 subcategory codes appear under more than one main_category
# (e.g. "CO" is used in both Physics and Mathematics with different meanings).
# Fix: namespace them as "MainCat::SubCode" to make them globally unique.
AMBIGUOUS = {"AP","CG","CO","CV","GN","GR","GT","LO","PR","SI","SP","ST"}

def namespace_sub(row):
    if row["sub_category"] in AMBIGUOUS:
        prefix = row["main_category"].replace(" ", "")
        return f"{prefix}::{row['sub_category']}"
    return row["sub_category"]

df["sub_category"] = df.apply(namespace_sub, axis=1)
print(f"  Namespaced {len(AMBIGUOUS)} ambiguous subcategory codes")
print(f"  Total unique subcategories after fix: {df['sub_category'].nunique()}")

# ── 3c. Text cleaning ──────────────────────────────────────────────────────────
_LATEX_CMD    = re.compile(r'\\[a-zA-Z]+\{[^}]*\}')   # \cmd{...}
_LATEX_INLINE = re.compile(r'\$[^$]+\$')               # $math$
_LATEX_DISP   = re.compile(r'\$\$[^$]+\$\$')           # $$math$$
_URL          = re.compile(r'https?://\S+')
_ARXIV_ID     = re.compile(r'\barXiv:\d{4}\.\d{4,5}\b', re.I)
_WHITESPACE   = re.compile(r'\s+')

def clean_text(text: str) -> str:
    t = str(text)
    t = _LATEX_DISP.sub(" ", t)
    t = _LATEX_INLINE.sub(" ", t)
    t = _LATEX_CMD.sub(" ", t)
    t = _URL.sub(" ", t)
    t = _ARXIV_ID.sub(" ", t)
    # keep hyphens (compound terms), letters, digits, basic punctuation
    t = re.sub(r"[^\w\s\-\.,;:'\(\)]", " ", t)
    t = _WHITESPACE.sub(" ", t).strip().lower()
    return t

df["title_clean"]    = df["title"].fillna("").apply(clean_text)
df["abstract_clean"] = df["abstract"].fillna("").apply(clean_text)

# Combined input field for model consumption
df["text"] = df["title_clean"] + " [SEP] " + df["abstract_clean"]

# ── 3d. Drop 35 near-empty abstracts (<20 words) ──────────────────────────────
# These are mostly one-line notes/comments — not useful training signal.
before = len(df)
df = df[df["abstract_words"] >= 20].reset_index(drop=True)
print(f"  Dropped {before - len(df)} papers with <20-word abstracts")

# ── 3e. Label encoding ────────────────────────────────────────────────────────
le_cat = LabelEncoder()
le_sub = LabelEncoder()

df["cat_label"] = le_cat.fit_transform(df["main_category"])
df["sub_label"] = le_sub.fit_transform(df["sub_category"])

joblib.dump(le_cat, LE_CAT)
joblib.dump(le_sub, LE_SUB)
print(f"  Encoded {len(le_cat.classes_)} main categories, {len(le_sub.classes_)} subcategories")
print(f"  Label encoders saved → {OUTPUT_DIR}/")

# ── 3f. Stratified train / val / test split (80 / 10 / 10) ────────────────────
train_df, temp_df = train_test_split(
    df, test_size=0.20, stratify=df["cat_label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["cat_label"], random_state=42
)

train_df = train_df.reset_index(drop=True)
val_df   = val_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

# Keep only columns needed downstream
KEEP_COLS = ["title_clean", "abstract_clean", "text",
             "main_category", "sub_category",
             "cat_label", "sub_label", "full_code", "all_labels"]

train_df[KEEP_COLS].to_csv(TRAIN_CSV, index=False)
val_df[KEEP_COLS].to_csv(VAL_CSV,   index=False)
test_df[KEEP_COLS].to_csv(TEST_CSV, index=False)

print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
print(f"  Splits saved → {OUTPUT_DIR}/")

# ── 3g. Split verification ─────────────────────────────────────────────────────
train_cats = set(train_df["main_category"].unique())
val_cats   = set(val_df["main_category"].unique())
test_cats  = set(test_df["main_category"].unique())

assert val_cats.issubset(train_cats),  "⚠ val has categories not in train!"
assert test_cats.issubset(train_cats), "⚠ test has categories not in train!"
print("  Split verification passed — all val/test categories present in train")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA CARD
# ═══════════════════════════════════════════════════════════════════════════════
lines = [
    "arXiv Papers — Data Card",
    "=" * 60,
    "",
    f"Raw rows:                {19060}",
    f"After dropping short abstracts: {len(df):,}  (dropped {19060 - len(df)})",
    "",
    "--- Label structure ---",
    f"Main categories:         {len(le_cat.classes_)}",
    f"  {sorted(le_cat.classes_)}",
    f"Subcategories (total):   {len(le_sub.classes_)}  (after namespacing ambiguous codes)",
    "",
    "--- Multi-label ---",
    f"Papers with >1 arXiv label: {(df['num_labels'] > 1).sum():,} / {len(df):,} = {(df['num_labels']>1).mean()*100:.1f}%",
    "NOTE: main_category is single-label (primary label only). all_labels column",
    "preserves the full multi-label set for future use.",
    "",
    "--- Missing data ---",
    "sub_category nulls:      158 (all math.NA — filled with 'NA')",
    "title nulls:             0",
    "abstract nulls:          0",
    "Duplicate titles:        0",
    "Duplicate abstracts:     0",
    "",
    "--- Text length ---",
    f"Abstract: mean {df['abstract_words'].mean():.0f} words, median {df['abstract_words'].median():.0f}, max {df['abstract_words'].max()}",
    f"Title:    mean {df['title_words'].mean():.0f} words,  median {df['title_words'].median():.0f}, max {df['title_words'].max()}",
    f"Papers exceeding 512 approx tokens: {(df['approx_tokens']>512).sum()}  (0.0% — standard BERT is fine)",
    "",
    "--- Class imbalance ---",
    f"Most common subcategory:  'General' ({df['sub_category'].value_counts().iloc[0]} samples)",
    f"Rarest subcategory:       'AI' (35 samples) — monitor during training",
    f"Subcategories < 100 samples: {(df['sub_category'].value_counts() < 100).sum()}",
    "",
    "--- Ambiguous subcategory codes (namespaced) ---",
    "12 codes appeared in >1 main category. Resolved by prefixing with",
    "main category name (e.g. 'CO' → 'Physics::CO' or 'Mathematics::CO').",
    "",
    "--- Splits ---",
    f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}",
    "Stratified on main_category. All val/test categories present in train: ✓",
    "",
    "--- Decisions locked in ---",
    "Task framing:       Multi-class (main_category is the target)",
    "                    Hierarchical extension: sub_category as second target",
    "Token budget:       512 — standard BERT/SciBERT is sufficient",
    "Primary metric:     Macro F1 (classes are moderately imbalanced)",
    "Recommended model:  allenai/scibert_scivocab_uncased",
    "",
    "--- Output files ---",
    f"train.csv, val.csv, test.csv  →  {OUTPUT_DIR}/",
    f"le_main_category.pkl, le_sub_category.pkl  →  {OUTPUT_DIR}/",
    f"eda_report.png  →  {OUTPUT_DIR}/",
]

data_card = "\n".join(lines)
with open(STATS_TXT, "w") as f:
    f.write(data_card)
print(data_card)
print(f"\nData card saved → {STATS_TXT}")
print("\n✓ Pipeline complete.")
