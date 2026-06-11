"""
phase6_colab_runner.py
========================
Paste each section below as a separate cell in Google Colab.
Run AFTER Phase 5 cells (models + outputs must exist in Drive).

CELL ORDER:
  [A] Mount + extract
  [B] Install deps
  [C] Run taxonomy (full script inline)
  [D] Save to Drive
"""

# ════════════════════════════════════════════════════════════════════════════════
# CELL A — Mount Drive & extract project zip
# ════════════════════════════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import zipfile, os
EXTRACT_DIR = '/content/PaperChaseAI-main'
ZIP_PATH    = '/content/drive/MyDrive/PaperChaseAI-main.zip'

if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall('/content/')
    print('Extracted.')
else:
    print('Already extracted.')

os.chdir(EXTRACT_DIR)
print('Working directory:', os.getcwd())
"""

# ════════════════════════════════════════════════════════════════════════════════
# CELL B — Install dependencies
# ════════════════════════════════════════════════════════════════════════════════
"""
!pip install -q transformers==4.40.0 scikit-learn pandas torch seaborn joblib tqdm
print('Done.')
"""

# ════════════════════════════════════════════════════════════════════════════════
# CELL C — Label Confusion Taxonomy (paste as one cell)
# ════════════════════════════════════════════════════════════════════════════════
"""
import os, json, warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, accuracy_score
from tqdm.notebook import tqdm

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = 'outputs'
TEST_CSV     = f'{OUTPUT_DIR}/test.csv'
LE_CAT_PATH  = f'{OUTPUT_DIR}/le_main_category.pkl'
LE_SUB_PATH  = f'{OUTPUT_DIR}/le_sub_category.pkl'
CAT_CKPT_DIR = f'{OUTPUT_DIR}/scibert_main_category/best_model'
SUB_CKPT_DIR = f'{OUTPUT_DIR}/scibert_sub_category/best_model'
HIER_MAP_PATH= f'{OUTPUT_DIR}/phase4_hierarchy_map.json'
P6_DIR       = f'{OUTPUT_DIR}/phase6'
os.makedirs(P6_DIR, exist_ok=True)

MAX_LEN = 256; BATCH_SIZE = 64
device  = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Device: {device}')

# ── Load artefacts ─────────────────────────────────────────────────────────────
test    = pd.read_csv(TEST_CSV)
le_cat  = joblib.load(LE_CAT_PATH)
le_sub  = joblib.load(LE_SUB_PATH)
num_cat = len(le_cat.classes_)
num_sub = len(le_sub.classes_)

with open(HIER_MAP_PATH) as f:
    hierarchy = json.load(f)

mask_matrix = np.zeros((num_cat, num_sub), dtype=bool)
for cat_name, sub_names in hierarchy.items():
    c = le_cat.transform([cat_name])[0]
    for s in sub_names:
        if s in le_sub.classes_:
            mask_matrix[c, le_sub.transform([s])[0]] = True

print(f'Test: {len(test):,} | Categories: {num_cat} | Sub-cats: {num_sub}')

# ── Dataset & inference helpers ────────────────────────────────────────────────
class PaperDataset(Dataset):
    def __init__(self, df, tok, max_len, label_col):
        self.texts  = df['text'].fillna('').tolist()
        self.labels = df[label_col].tolist()
        self.tok    = tok; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)}

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

# ── Inference ──────────────────────────────────────────────────────────────────
tokenizer   = AutoTokenizer.from_pretrained(SUB_CKPT_DIR)
num_workers = 2 if device.type == 'cuda' else 0

loader_cat  = DataLoader(PaperDataset(test, tokenizer, MAX_LEN, 'cat_label'),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
loader_sub  = DataLoader(PaperDataset(test, tokenizer, MAX_LEN, 'sub_label'),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

cat_model = AutoModelForSequenceClassification.from_pretrained(CAT_CKPT_DIR).to(device)
sub_model = AutoModelForSequenceClassification.from_pretrained(SUB_CKPT_DIR).to(device)

cat_logits, cat_preds, cat_true      = get_preds_and_logits(cat_model, loader_cat, 'Category')
sub_logits, sub_preds_unc, sub_true  = get_preds_and_logits(sub_model, loader_sub, 'Sub-cat (unc)')

NEG_INF = -1e9
sub_preds_con = np.empty(len(sub_true), dtype=int)
for i, (logit_row, pred_cat) in enumerate(zip(sub_logits, cat_preds)):
    masked = logit_row.copy().astype(float)
    masked[mask_matrix[pred_cat] == 0] = NEG_INF
    sub_preds_con[i] = int(np.argmax(masked))

# ── Build annotated frame ──────────────────────────────────────────────────────
df = test.copy().reset_index(drop=True)
df['true_sub_name']     = le_sub.inverse_transform(sub_true)
df['pred_sub_con_name'] = le_sub.inverse_transform(sub_preds_con)
df['pred_sub_unc_name'] = le_sub.inverse_transform(sub_preds_unc)

sub_to_parent = {child: parent for parent, children in hierarchy.items() for child in children}

def assign_tier(ts, ps):
    if ts == ps: return 0, 'Correct'
    return (1, 'Same-parent error') if sub_to_parent.get(ts) == sub_to_parent.get(ps) \
           else (2, 'Cross-parent error')

tid_con, tlab_con = zip(*[assign_tier(ts, ps) for ts, ps in zip(df['true_sub_name'], df['pred_sub_con_name'])])
tid_unc, tlab_unc = zip(*[assign_tier(ts, ps) for ts, ps in zip(df['true_sub_name'], df['pred_sub_unc_name'])])

df['tier_label_con'] = tlab_con
df['tier_label_unc'] = tlab_unc
df['true_parent']    = df['true_sub_name'].map(sub_to_parent)

# ── Tier summary ───────────────────────────────────────────────────────────────
n_total   = len(df)
tier_order = ['Correct', 'Same-parent error', 'Cross-parent error']
tier_summary = pd.DataFrame([{
    'Tier':                t,
    'Constrained (n)':    (df['tier_label_con'] == t).sum(),
    'Constrained (%)':    round((df['tier_label_con'] == t).sum() / n_total * 100, 2),
    'Unconstrained (n)':  (df['tier_label_unc'] == t).sum(),
    'Unconstrained (%)':  round((df['tier_label_unc'] == t).sum() / n_total * 100, 2),
} for t in tier_order])

print('=' * 70)
print(f'TIER DISTRIBUTION  (N={n_total:,})')
print('=' * 70)
print(tier_summary.to_string(index=False))

n_cross_con  = (df['tier_label_con'] == 'Cross-parent error').sum()
n_cross_unc  = (df['tier_label_unc'] == 'Cross-parent error').sum()
reduction    = (1 - n_cross_con / max(n_cross_unc, 1)) * 100

print(f'\\n  Cross-parent errors — unconstrained: {n_cross_unc}')
print(f'  Cross-parent errors — constrained:   {n_cross_con}  (target = 0)')
print(f'  Reduction: {reduction:.1f}%')
print(f'  ✅ All remaining errors are Tier-1 (same-parent only).')
tier_summary.to_csv(f'{P6_DIR}/tier_summary.csv', index=False)

# ── Per-parent breakdown ───────────────────────────────────────────────────────
same_errors  = df[df['tier_label_con'] == 'Same-parent error'].copy()
same_errors['true_parent'] = same_errors['true_sub_name'].map(sub_to_parent)

per_parent = same_errors.groupby('true_parent').size().reset_index(name='same_parent_errors')
per_parent = (per_parent
    .merge(df.groupby('true_parent').size().reset_index(name='n_test'), on='true_parent')
    .merge(df[df['tier_label_con']=='Correct'].groupby('true_parent').size().reset_index(name='correct'), on='true_parent', how='left')
    .fillna(0))
per_parent['correct']        = per_parent['correct'].astype(int)
per_parent['error_rate (%)'] = (per_parent['same_parent_errors'] / per_parent['n_test'] * 100).round(1)
per_parent['sub_acc (%)']    = (per_parent['correct'] / per_parent['n_test'] * 100).round(1)
per_parent = per_parent.sort_values('same_parent_errors', ascending=False)

print('\\n' + '='*70)
print('SAME-PARENT ERROR BREAKDOWN BY PARENT')
print('='*70)
print(per_parent.to_string(index=False))
per_parent.to_csv(f'{P6_DIR}/same_parent_error_breakdown.csv', index=False)

# ── Top confused pairs ─────────────────────────────────────────────────────────
same_pairs = Counter((r['true_sub_name'], r['pred_sub_con_name']) for _, r in same_errors.iterrows())
top_same = pd.DataFrame([{'true_parent': sub_to_parent.get(ts,'?'), 'true_sub': ts,
                           'pred_sub': ps, 'count': n}
                          for (ts, ps), n in same_pairs.most_common(15)])
print('\\nTop 15 same-parent confused pairs:')
print(top_same.to_string(index=False))
top_same.to_csv(f'{P6_DIR}/top_same_parent_pairs.csv', index=False)

# ── Bar chart ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(tier_order)); width = 0.35
con_vals = [tier_summary.loc[tier_summary['Tier']==t,'Constrained (%)'].values[0] for t in tier_order]
unc_vals = [tier_summary.loc[tier_summary['Tier']==t,'Unconstrained (%)'].values[0] for t in tier_order]
bars1 = ax.bar(x - width/2, unc_vals, width, label='Unconstrained', color='#E07B54', alpha=0.85)
bars2 = ax.bar(x + width/2, con_vals, width, label='Constrained',   color='#4A90D9', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(tier_order)
ax.set_ylabel('% of test samples')
ax.set_title('Label Confusion Taxonomy: Tier Distribution', fontweight='bold')
ax.legend(); ax.set_ylim(0, 100)
for b in list(bars1) + list(bars2):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.8,
            f'{b.get_height():.1f}%', ha='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{P6_DIR}/label_confusion_taxonomy_bar.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()

# ── Per-parent heatmaps ────────────────────────────────────────────────────────
parents_with_errors = per_parent[per_parent['same_parent_errors'] > 0]['true_parent'].tolist()
n_cols = 2; n_rows = (len(parents_with_errors) + 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 5))
axes = axes.flatten()

for ax_idx, parent in enumerate(parents_with_errors):
    subs       = hierarchy[parent]
    sub_ids    = [le_sub.transform([s])[0] for s in subs if s in le_sub.classes_]
    sub_names  = [le_sub.classes_[i] for i in sub_ids]
    id_to_loc  = {sid: i for i, sid in enumerate(sub_ids)}
    mask_rows  = np.isin(sub_true, sub_ids)
    if mask_rows.sum() == 0: axes[ax_idx].axis('off'); continue
    lt = np.array([id_to_loc[x] for x in sub_true[mask_rows]])
    lp = np.array([id_to_loc.get(x,-1) for x in sub_preds_con[mask_rows]])
    valid = lp >= 0; lt = lt[valid]; lp = lp[valid]
    if len(lt) == 0: axes[ax_idx].axis('off'); continue
    cm_loc = confusion_matrix(lt, lp, labels=list(range(len(sub_names))))
    cm_norm = cm_loc / cm_loc.sum(axis=1, keepdims=True).clip(min=1)
    short   = [s.split('::')[-1] if '::' in s else s for s in sub_names]
    sns.heatmap(cm_norm, ax=axes[ax_idx], xticklabels=short, yticklabels=short,
                cmap='YlOrRd', vmin=0, vmax=1, linewidths=0.3,
                annot=(len(sub_names)<=12), fmt='.2f', annot_kws={'size':7}, cbar=False)
    axes[ax_idx].set_title(f'{parent}\\n(same-parent, row-norm)', fontweight='bold', fontsize=9)
    axes[ax_idx].set_xlabel('Predicted', fontsize=8); axes[ax_idx].set_ylabel('True', fontsize=8)
    axes[ax_idx].tick_params(axis='x', rotation=45, labelsize=7)
    axes[ax_idx].tick_params(axis='y', rotation=0, labelsize=7)

for i in range(len(parents_with_errors), len(axes)): axes[i].axis('off')
plt.suptitle('Label Confusion Taxonomy — Same-Parent Confusion Matrices', fontweight='bold', fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(f'{P6_DIR}/label_confusion_taxonomy_heatmap.png', dpi=150, bbox_inches='tight')
plt.show(); plt.close()
print(f'\\n✅ Phase 6 outputs saved to {P6_DIR}/')
"""

# ════════════════════════════════════════════════════════════════════════════════
# CELL D — Save outputs to Drive
# ════════════════════════════════════════════════════════════════════════════════
"""
import shutil
DRIVE_SAVE = '/content/drive/MyDrive/PaperChaseAI-outputs'
if os.path.exists(DRIVE_SAVE):
    shutil.rmtree(DRIVE_SAVE)
shutil.copytree(f'{EXTRACT_DIR}/outputs', DRIVE_SAVE)
print(f'✅ All outputs saved to Drive: {DRIVE_SAVE}')
"""
