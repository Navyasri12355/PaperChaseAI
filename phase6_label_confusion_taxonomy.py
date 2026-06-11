"""
phase6_label_confusion_taxonomy.py
===================================
Label Confusion Taxonomy — Phase 6 analysis for PaperChaseAI.

Classifies every sub-category misclassification into one of three tiers:
  Tier 0 – Correct
  Tier 1 – Same-parent error   (true & predicted sub share the same parent)
  Tier 2 – Cross-parent error  (true & predicted sub belong to DIFFERENT parents)

The constrained model eliminates Tier-2 errors by design.
This script proves that claim quantitatively and produces:
  outputs/phase6/tier_summary.csv
  outputs/phase6/same_parent_error_breakdown.csv
  outputs/phase6/top_same_parent_pairs.csv
  outputs/phase6/label_confusion_taxonomy_heatmap.png
  outputs/phase6/label_confusion_taxonomy_report.txt

Run from the project root (same level as outputs/):
    python phase6_label_confusion_taxonomy.py

Prerequisites: phase 4 + phase 5 must have been run so that
  outputs/test.csv, outputs/le_*.pkl, outputs/phase4_hierarchy_map.json,
  outputs/scibert_main_category/best_model/  and
  outputs/scibert_sub_category/best_model/   all exist.
"""

# ── 0. Imports ─────────────────────────────────────────────────────────────────
import os
import json
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ── 1. Paths ───────────────────────────────────────────────────────────────────
OUTPUT_DIR    = 'outputs'
TEST_CSV      = f'{OUTPUT_DIR}/test.csv'
LE_CAT_PATH   = f'{OUTPUT_DIR}/le_main_category.pkl'
LE_SUB_PATH   = f'{OUTPUT_DIR}/le_sub_category.pkl'
CAT_CKPT_DIR  = f'{OUTPUT_DIR}/scibert_main_category/best_model'
SUB_CKPT_DIR  = f'{OUTPUT_DIR}/scibert_sub_category/best_model'
HIER_MAP_PATH = f'{OUTPUT_DIR}/phase4_hierarchy_map.json'
P6_DIR        = f'{OUTPUT_DIR}/phase6'
os.makedirs(P6_DIR, exist_ok=True)

MAX_LEN    = 256
BATCH_SIZE = 64

device = (
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cpu')
)
print(f'Device: {device}')

# ── 2. Load artefacts ──────────────────────────────────────────────────────────
test      = pd.read_csv(TEST_CSV)
le_cat    = joblib.load(LE_CAT_PATH)
le_sub    = joblib.load(LE_SUB_PATH)
num_cat   = len(le_cat.classes_)
num_sub   = len(le_sub.classes_)

with open(HIER_MAP_PATH) as f:
    hierarchy = json.load(f)

# Build hierarchical mask: mask_matrix[cat_id, sub_id] = True if valid
mask_matrix = np.zeros((num_cat, num_sub), dtype=bool)
for cat_name, sub_names in hierarchy.items():
    c = le_cat.transform([cat_name])[0]
    for s_name in sub_names:
        if s_name in le_sub.classes_:
            mask_matrix[c, le_sub.transform([s_name])[0]] = True

print(f'Test: {len(test):,}  |  Categories: {num_cat}  Sub-cats: {num_sub}')

# ── 3. Dataset & inference helpers ────────────────────────────────────────────
class PaperDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label_col):
        self.texts   = df['text'].fillna('').tolist()
        self.labels  = df[label_col].tolist()
        self.tok     = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_preds_and_logits(model, loader, desc='Inference'):
    model.eval()
    logits_list, preds_list, labels_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            out  = model(input_ids=ids, attention_mask=mask)
            logits_list.append(out.logits.cpu().numpy())
            preds_list.extend(out.logits.argmax(-1).cpu().numpy())
            labels_list.extend(batch['label'].numpy())
    return np.vstack(logits_list), np.array(preds_list), np.array(labels_list)


# ── 4. Run inference ───────────────────────────────────────────────────────────
print('\nLoading models and running inference on test set...')
tokenizer = AutoTokenizer.from_pretrained(SUB_CKPT_DIR)
num_workers = 2 if device.type == 'cuda' else 0

loader_cat = DataLoader(
    PaperDataset(test, tokenizer, MAX_LEN, 'cat_label'),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
)
loader_sub = DataLoader(
    PaperDataset(test, tokenizer, MAX_LEN, 'sub_label'),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers
)

cat_model = AutoModelForSequenceClassification.from_pretrained(CAT_CKPT_DIR).to(device)
sub_model = AutoModelForSequenceClassification.from_pretrained(SUB_CKPT_DIR).to(device)

cat_logits, cat_preds, cat_true   = get_preds_and_logits(cat_model, loader_cat, 'Category model')
sub_logits, sub_preds_unc, sub_true = get_preds_and_logits(sub_model, loader_sub, 'Sub-cat model (unc)')

# Apply hierarchical constraint
NEG_INF = -1e9
sub_preds_con = np.empty(len(sub_true), dtype=int)
for i, (logit_row, pred_cat) in enumerate(zip(sub_logits, cat_preds)):
    masked = logit_row.copy().astype(float)
    masked[mask_matrix[pred_cat] == 0] = NEG_INF
    sub_preds_con[i] = int(np.argmax(masked))

print(f'  Cat acc:  {accuracy_score(cat_true, cat_preds):.4f}')
print(f'  Sub acc (constrained):   {accuracy_score(sub_true, sub_preds_con):.4f}')
print(f'  Sub acc (unconstrained): {accuracy_score(sub_true, sub_preds_unc):.4f}')

# ── 5. Build annotated test frame ──────────────────────────────────────────────
df = test.copy().reset_index(drop=True)
df['true_sub_name']      = le_sub.inverse_transform(sub_true)
df['pred_sub_con_name']  = le_sub.inverse_transform(sub_preds_con)
df['pred_sub_unc_name']  = le_sub.inverse_transform(sub_preds_unc)

# ── 6. Tier assignment ─────────────────────────────────────────────────────────
# Build sub -> parent lookup
sub_to_parent = {}
for parent, children in hierarchy.items():
    for child in children:
        sub_to_parent[child] = parent


def assign_tier(true_sub_name: str, pred_sub_name: str) -> tuple:
    """Return (tier_id, tier_label) for a single prediction."""
    if true_sub_name == pred_sub_name:
        return 0, 'Correct'
    true_parent = sub_to_parent.get(true_sub_name, '__unknown__')
    pred_parent = sub_to_parent.get(pred_sub_name, '__unknown__')
    if true_parent == pred_parent:
        return 1, 'Same-parent error'
    return 2, 'Cross-parent error'


# Constrained model tiers
tier_ids_con, tier_labels_con = zip(*[
    assign_tier(ts, ps)
    for ts, ps in zip(df['true_sub_name'], df['pred_sub_con_name'])
])
df['tier_id_con']    = tier_ids_con
df['tier_label_con'] = tier_labels_con

# Unconstrained model tiers
tier_ids_unc, tier_labels_unc = zip(*[
    assign_tier(ts, ps)
    for ts, ps in zip(df['true_sub_name'], df['pred_sub_unc_name'])
])
df['tier_id_unc']    = tier_ids_unc
df['tier_label_unc'] = tier_labels_unc

# Flag rows where the category model was correct
# Constrained cross-parent errors = cat model errors (propagated to sub)
# Unconstrained cross-parent errors = sub model's own cross-parent mistakes
# These two sources happen to be similar in size (~16% each).
# The correct comparison is CONDITIONAL on cat being correct.
df['cat_correct']  = (cat_preds == cat_true)
df['true_parent']  = df['true_sub_name'].map(sub_to_parent)

# Save full annotated frame — consumed by phase7_semantic_similarity.py
df.to_csv(f'{P6_DIR}/all_predictions.csv', index=False)
print(f'  Predictions saved -> {P6_DIR}/all_predictions.csv')

# ── 7. Tier summary tables ────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('TIER DISTRIBUTION — OVERALL (test set, N={:,})'.format(len(df)))
print('=' * 70)

n_total    = len(df)
tier_order = ['Correct', 'Same-parent error', 'Cross-parent error']

tier_summary = pd.DataFrame([
    {
        'Tier':                 t,
        'Constrained (n)':     (df['tier_label_con'] == t).sum(),
        'Constrained (%)':     round((df['tier_label_con'] == t).sum() / n_total * 100, 2),
        'Unconstrained (n)':   (df['tier_label_unc'] == t).sum(),
        'Unconstrained (%)':   round((df['tier_label_unc'] == t).sum() / n_total * 100, 2),
    }
    for t in tier_order
])
print(tier_summary.to_string(index=False))

n_cross_con = (df['tier_label_con'] == 'Cross-parent error').sum()
n_cross_unc = (df['tier_label_unc'] == 'Cross-parent error').sum()
n_cat_err   = (~df['cat_correct']).sum()
reduction_pct = (n_cross_unc - n_cross_con) / n_cross_unc * 100 if n_cross_unc > 0 else 0.0

print(f'\n  NOTE: Constrained cross-parent errors ({n_cross_con}) ≈ category model errors ({n_cat_err})')
print(f'        because when cat_model is wrong, the mask forces sub into the wrong parent.')
print(f'        Unconstrained cross-parent errors ({n_cross_unc}) are the sub-model\'s own mistakes.')
print(f'        These two sources happen to be similar -> overall bars look identical.')
print(f'        The correct comparison is CONDITIONAL on category model being correct (see below).')

# ── Conditional tier summary: given cat_correct = True ────────────────────────
df_cat_ok  = df[df['cat_correct']].copy()
n_cat_ok   = len(df_cat_ok)
print('\n' + '=' * 70)
print(f'TIER DISTRIBUTION — CONDITIONAL ON CORRECT CATEGORY (N={n_cat_ok:,})')
print('=' * 70)
print('  When the category model is correct, the constraint GUARANTEES zero cross-parent sub-errors.')
print('  This is the clean NLP argument for the hierarchical constraint.')

tier_cond = pd.DataFrame([
    {
        'Tier':                 t,
        'Constrained (n)':     (df_cat_ok['tier_label_con'] == t).sum(),
        'Constrained (%)':     round((df_cat_ok['tier_label_con'] == t).sum() / n_cat_ok * 100, 2),
        'Unconstrained (n)':   (df_cat_ok['tier_label_unc'] == t).sum(),
        'Unconstrained (%)':   round((df_cat_ok['tier_label_unc'] == t).sum() / n_cat_ok * 100, 2),
    }
    for t in tier_order
])
print(tier_cond.to_string(index=False))

n_cross_con_cond = (df_cat_ok['tier_label_con'] == 'Cross-parent error').sum()
n_cross_unc_cond = (df_cat_ok['tier_label_unc'] == 'Cross-parent error').sum()
print(f'\n  -> Constrained cross-parent (given cat correct): {n_cross_con_cond}  ← ZERO by design')
print(f'  -> Unconstrained cross-parent (given cat correct): {n_cross_unc_cond}  ← sub-model\'s own errors')

tier_summary.to_csv(f'{P6_DIR}/tier_summary.csv', index=False)
tier_cond.to_csv(f'{P6_DIR}/tier_summary_conditional.csv', index=False)
print(f'  Saved -> {P6_DIR}/tier_summary.csv  +  tier_summary_conditional.csv')

# ── 8. Per-parent same-parent breakdown ───────────────────────────────────────
print('\n' + '=' * 70)
print('SAME-PARENT ERROR BREAKDOWN BY PARENT CATEGORY')
print('=' * 70)

same_errors = df[df['tier_label_con'] == 'Same-parent error'].copy()
same_errors['true_parent'] = same_errors['true_sub_name'].map(sub_to_parent)

df['true_parent'] = df['true_sub_name'].map(sub_to_parent)

per_parent = (
    same_errors
    .groupby('true_parent')
    .size()
    .reset_index(name='same_parent_errors')
)
parent_totals  = df.groupby('true_parent').size().reset_index(name='n_test')
parent_correct = (
    df[df['tier_label_con'] == 'Correct']
    .groupby('true_parent')
    .size()
    .reset_index(name='correct')
)

per_parent = (
    per_parent
    .merge(parent_totals, on='true_parent', how='left')
    .merge(parent_correct, on='true_parent', how='left')
    .fillna(0)
)
per_parent['correct']         = per_parent['correct'].astype(int)
per_parent['error_rate (%)']  = (per_parent['same_parent_errors'] / per_parent['n_test'] * 100).round(1)
per_parent['sub_acc (%)']     = (per_parent['correct'] / per_parent['n_test'] * 100).round(1)
per_parent = per_parent.sort_values('same_parent_errors', ascending=False)

print(per_parent.to_string(index=False))
per_parent.to_csv(f'{P6_DIR}/same_parent_error_breakdown.csv', index=False)
print(f'  Saved -> {P6_DIR}/same_parent_error_breakdown.csv')

# ── 9. Top confused same-parent pairs ─────────────────────────────────────────
print('\n  Top 15 same-parent confused sub-category pairs (constrained model):')
print('  ' + '-' * 60)

same_pairs = Counter()
for _, row in same_errors.iterrows():
    same_pairs[(row['true_sub_name'], row['pred_sub_con_name'])] += 1

top_same = pd.DataFrame([
    {
        'true_parent': sub_to_parent.get(ts, '?'),
        'true_sub':    ts,
        'pred_sub':    ps,
        'count':       n,
    }
    for (ts, ps), n in same_pairs.most_common(15)
])
print(top_same.to_string(index=False))
top_same.to_csv(f'{P6_DIR}/top_same_parent_pairs.csv', index=False)
print(f'  Saved -> {P6_DIR}/top_same_parent_pairs.csv')

# ── 10. Heatmaps: same-parent confusion within each parent ────────────────────
print('\n  Generating per-parent confusion heatmaps...')

parents_with_errors = (
    per_parent[per_parent['same_parent_errors'] > 0]['true_parent'].tolist()
)

n_cols = 2
n_rows = (len(parents_with_errors) + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
axes = axes.flatten()

for ax_idx, parent in enumerate(parents_with_errors):
    subs     = hierarchy[parent]
    sub_ids  = [le_sub.transform([s])[0] for s in subs if s in le_sub.classes_]
    sub_names = [le_sub.classes_[i] for i in sub_ids]
    id_to_local = {sid: i for i, sid in enumerate(sub_ids)}

    # Filter test rows whose true label belongs to this parent
    mask_rows  = np.isin(sub_true, sub_ids)
    if mask_rows.sum() == 0:
        axes[ax_idx].axis('off')
        continue

    local_true = np.array([id_to_local[x] for x in sub_true[mask_rows]])
    local_pred = np.array([id_to_local.get(x, -1) for x in sub_preds_con[mask_rows]])

    valid      = local_pred >= 0
    local_true = local_true[valid]
    local_pred = local_pred[valid]

    if len(local_true) == 0:
        axes[ax_idx].axis('off')
        continue

    cm_local = confusion_matrix(
        local_true, local_pred, labels=list(range(len(sub_names)))
    )
    row_sums = cm_local.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm_local / row_sums          # row-normalised -> recall per class

    short_names = [s.split('::')[-1] if '::' in s else s for s in sub_names]
    annot_data  = cm_norm if len(sub_names) <= 12 else False

    sns.heatmap(
        cm_norm, ax=axes[ax_idx],
        xticklabels=short_names, yticklabels=short_names,
        cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.3,
        annot=annot_data, fmt='.2f',
        annot_kws={'size': 7}, cbar=False,
    )
    axes[ax_idx].set_title(
        f'{parent}\n(same-parent, row-normalised recall)',
        fontweight='bold', fontsize=9
    )
    axes[ax_idx].set_xlabel('Predicted sub-cat', fontsize=8)
    axes[ax_idx].set_ylabel('True sub-cat', fontsize=8)
    axes[ax_idx].tick_params(axis='x', rotation=45, labelsize=7)
    axes[ax_idx].tick_params(axis='y', rotation=0,  labelsize=7)

for i in range(len(parents_with_errors), len(axes)):
    axes[i].axis('off')

plt.suptitle(
    'Label Confusion Taxonomy — Same-Parent Confusion Matrices (Constrained Model)',
    fontweight='bold', fontsize=11, y=1.01
)
plt.tight_layout()
heatmap_path = f'{P6_DIR}/label_confusion_taxonomy_heatmap.png'
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> {heatmap_path}')

# ── 11. Two-panel bar chart: overall + conditional ────────────────────────────
# Left panel: overall (shows why cross-parent looks the same — different sources)
# Right panel: conditional on cat correct (shows constraint eliminates cross-parent)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

x     = np.arange(len(tier_order))
width = 0.35
short_labels = ['Correct', 'Same-parent\nerror', 'Cross-parent\nerror']

# ── Left panel: overall ───────────────────────────────────────────────────────
con_vals_all = [tier_summary.loc[tier_summary['Tier'] == t, 'Constrained (%)'].values[0]  for t in tier_order]
unc_vals_all = [tier_summary.loc[tier_summary['Tier'] == t, 'Unconstrained (%)'].values[0] for t in tier_order]

bars1 = ax1.bar(x - width/2, unc_vals_all, width, label='Unconstrained', color='#E07B54', alpha=0.85)
bars2 = ax1.bar(x + width/2, con_vals_all, width, label='Constrained',   color='#4A90D9', alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(short_labels, fontsize=9)
ax1.set_ylabel('% of ALL test samples', fontsize=10)
ax1.set_title(
    'Overall Tier Distribution\n'
    f'(N={n_total:,} — includes category model errors)',
    fontweight='bold', fontsize=10
)
ax1.legend(fontsize=9); ax1.set_ylim(0, 80)
for bar in list(bars1) + list(bars2):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)
ax1.text(0.5, -0.22,
    f'Cross-parent bars are equal because constrained cross-parent errors\n'
    f'come from category model mistakes ({n_cat_err} samples), not sub-model failures.',
    transform=ax1.transAxes, ha='center', fontsize=8, color='#666666',
    style='italic', wrap=True)

# ── Right panel: conditional on category correct ──────────────────────────────
con_vals_cond = [tier_cond.loc[tier_cond['Tier'] == t, 'Constrained (%)'].values[0]  for t in tier_order]
unc_vals_cond = [tier_cond.loc[tier_cond['Tier'] == t, 'Unconstrained (%)'].values[0] for t in tier_order]

bars3 = ax2.bar(x - width/2, unc_vals_cond, width, label='Unconstrained', color='#E07B54', alpha=0.85)
bars4 = ax2.bar(x + width/2, con_vals_cond, width, label='Constrained',   color='#4A90D9', alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(short_labels, fontsize=9)
ax2.set_ylabel('% of cat-correct test samples', fontsize=10)
ax2.set_title(
    'Conditional Tier Distribution\n'
    f'(N={n_cat_ok:,} — given category model correct ✓)',
    fontweight='bold', fontsize=10
)
ax2.legend(fontsize=9); ax2.set_ylim(0, 80)
for bar in list(bars3) + list(bars4):
    val = bar.get_height()
    label = '0.0% ✓' if val == 0.0 else f'{val:.1f}%'
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5,
             label, ha='center', va='bottom', fontsize=8,
             fontweight='bold' if val == 0.0 else 'normal',
             color='#2ecc71' if val == 0.0 else 'black')
ax2.text(0.5, -0.22,
    f'When category is correct, the constraint eliminates ALL cross-parent\n'
    f'sub-category errors (constrained = 0, unconstrained = {n_cross_unc_cond}).',
    transform=ax2.transAxes, ha='center', fontsize=8, color='#2c7a2c',
    style='italic', wrap=True)

plt.suptitle(
    'Label Confusion Taxonomy: Hierarchical Constraint Effect',
    fontweight='bold', fontsize=12, y=1.02
)
plt.tight_layout()
bar_path = f'{P6_DIR}/label_confusion_taxonomy_bar.png'
plt.savefig(bar_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved -> {bar_path}')

# ── 12. Written report ─────────────────────────────────────────────────────────
sep = '=' * 70
report_lines = [
    sep,
    'PHASE 6 — LABEL CONFUSION TAXONOMY REPORT',
    sep,
    '',
    'STRATEGY: CONSTRAINED SEQUENTIAL (parent-masked logits)',
    '  The sub-category model logits are zeroed for any sub-category that',
    '  does NOT belong to the parent category predicted in step 1.',
    '  This forces every prediction to be taxonomically consistent.',
    '',
    'CLAIM UNDER TEST:',
    '  The masking constraint should reduce cross-parent (Tier-2) errors',
    '  to zero, leaving only same-parent (Tier-1) errors.',
    '',
    sep, 'TIER DEFINITIONS', sep,
    '  Tier 0 — Correct:           true_sub == pred_sub',
    '  Tier 1 — Same-parent error: true and pred share the same parent category',
    '  Tier 2 — Cross-parent error: true and pred belong to DIFFERENT parents',
    '',
    sep, 'TIER DISTRIBUTION', sep,
    tier_summary.to_string(index=False),
    '',
    f'  Cross-parent errors (unconstrained): {n_cross_unc}',
    f'  Cross-parent errors (constrained):   {n_cross_con}   ← target = 0',
    f'  Reduction: {reduction_pct:.1f}%',
    '',
    sep, 'CONDITIONAL TIER DISTRIBUTION (given cat model correct)', sep,
    tier_cond.to_string(index=False),
    '',
    f'  Constrained cross-parent (given cat correct): {n_cross_con_cond}  ← ZERO by design',
    f'  Unconstrained cross-parent (given cat correct): {n_cross_unc_cond}',
    '',
    sep, 'KEY FINDING', sep,
    '  WHY OVERALL CROSS-PARENT COUNTS ARE SIMILAR:',
    f'  Constrained cross-parent errors ({n_cross_con}) ≈ category model errors ({n_cat_err}).',
    '  When the category model predicts the wrong parent, the mask forces the sub',
    '  prediction into that wrong parent — propagating the error rather than fixing it.',
    '  Unconstrained cross-parent errors come from the sub-model making independent',
    f'  cross-parent mistakes ({n_cross_unc} total), which happen to be similar in scale.',
    '',
    '  THE CORRECT COMPARISON — CONDITIONAL ON CATEGORY CORRECT:',
    f'  Among the {n_cat_ok:,} samples where the category model is correct, the constraint',
    f'  reduces cross-parent sub-errors from {n_cross_unc_cond} (unconstrained) to 0 (constrained).',
    '  This is the definitive NLP argument: when the taxonomy root is identified correctly,',
    '  the constraint GUARANTEES all sub-category predictions stay within that branch.',
    '',
    '  Practical implication: A model confusing "LG" (Machine Learning) with',
    '  "CL" (Computation and Language) — both under Computer Science — is a',
    '  qualitatively less severe error than confusing them with "flu-dyn"',
    '  (Fluid Dynamics) under Physics. The constraint enforces the latter',
    '  guarantee for every sample where it correctly identifies the parent.',
    '',
    sep, 'PER-PARENT SAME-PARENT ERROR RATE', sep,
    per_parent.to_string(index=False),
    '',
    sep, 'TOP 15 SAME-PARENT CONFUSED PAIRS (constrained)', sep,
    top_same.to_string(index=False),
    '',
    sep, 'OUTPUTS', sep,
    f'  {P6_DIR}/tier_summary.csv',
    f'  {P6_DIR}/same_parent_error_breakdown.csv',
    f'  {P6_DIR}/top_same_parent_pairs.csv',
    f'  {P6_DIR}/label_confusion_taxonomy_heatmap.png',
    f'  {P6_DIR}/label_confusion_taxonomy_bar.png',
    f'  {P6_DIR}/label_confusion_taxonomy_report.txt',
]

report_text = '\n'.join(report_lines)
report_path = f'{P6_DIR}/label_confusion_taxonomy_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_text)

print('\n' + report_text)
print(f'\n✅  Phase 6 complete. All outputs in {P6_DIR}/')
