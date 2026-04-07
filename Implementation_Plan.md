# NLP Research Paper Classifier — Implementation Plan

## Overview

You have a CSV of ~23,000 research papers with `title`, `abstract`, `categories`, and `subcategories`. The goal is to build a classifier that can take a new paper (title + abstract) and predict its category and subcategory. This is a **multi-label / hierarchical text classification** problem.

---

## Phase 0 — Understand Your Data (Day 1)

Before writing a single model, spend time understanding what you're working with.

### 0.1 Exploratory Data Analysis (EDA)

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("papers.csv")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())

# Class distribution
df['category'].value_counts().plot(kind='bar', figsize=(14, 5))
df['subcategory'].value_counts().head(40).plot(kind='bar', figsize=(14, 5))
```

**Key questions to answer:**
- How many unique categories and subcategories exist?
- Is the distribution balanced or heavily skewed?
- Are there papers with multiple categories (multi-label)?
- What's the average abstract length?
- Are there missing abstracts (title-only papers)?

### 0.2 Decide on Task Framing

| Scenario | Framing |
|---|---|
| Each paper has exactly one category | Multi-class classification |
| Each paper can have multiple categories | Multi-label classification |
| You need to predict both category AND subcategory | Hierarchical classification |

---

## Phase 1 — Data Preparation (Day 1–2)

### 1.1 Cleaning

```python
def clean_text(text):
    import re
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\w\s\-]', '', text)     # remove special chars
    return text

df['text'] = (df['title'].fillna('') + ' [SEP] ' + df['abstract'].fillna('')).apply(clean_text)
```

**Why concatenate title + abstract?** The title gives a dense signal; the abstract provides context. Using `[SEP]` between them helps transformer models understand the boundary.

### 1.2 Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le_cat = LabelEncoder()
le_sub = LabelEncoder()

df['cat_label'] = le_cat.fit_transform(df['category'])
df['sub_label'] = le_sub.fit_transform(df['subcategory'])

# Save encoders for inference
import joblib
joblib.dump(le_cat, 'label_encoder_category.pkl')
joblib.dump(le_sub, 'label_encoder_subcategory.pkl')
```

### 1.3 Stratified Train/Val/Test Split

```python
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['cat_label'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['cat_label'], random_state=42)

# Sizes: ~18,400 train / ~2,300 val / ~2,300 test
```

---

## Phase 2 — Baseline Model (Day 2–3)

Always build a fast, dumb baseline before touching deep learning. It sets a floor to beat and catches data issues early.

### 2.1 TF-IDF + Logistic Regression

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, C=5, class_weight='balanced'))
])

pipeline.fit(train_df['text'], train_df['cat_label'])
preds = pipeline.predict(val_df['text'])
print(classification_report(val_df['cat_label'], preds, target_names=le_cat.classes_))
```

**Expected performance:** 70–85% accuracy depending on class balance. This is your baseline to beat.

### 2.2 Also Try: SVM and Gradient Boosting

```python
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# LinearSVC is often the best classical text classifier
svc = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, sublinear_tf=True)),
    ('clf', CalibratedClassifierCV(LinearSVC(max_iter=2000)))
])
```

---

## Phase 3 — Transformer-Based Model (Day 3–7)

This is where you'll get the best results. Use a pretrained language model fine-tuned for classification.

### 3.1 Model Choice

| Model | Why choose it |
|---|---|
| `allenai/scibert_scivocab_uncased` | Trained on scientific papers — **best choice for research paper classification** |
| `microsoft/BiomedNLP-PubMedBERT-base-uncased` | If many papers are biomedical |
| `bert-base-uncased` | General fallback, well understood |
| `distilbert-base-uncased` | Faster/smaller, slight accuracy tradeoff |

**Recommended: SciBERT** — its vocabulary was built from scientific text, so domain terms tokenize better.

### 3.2 Dataset Class

```python
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

class PaperDataset(Dataset):
    def __init__(self, texts, labels, max_len=512):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
```

### 3.3 Model Architecture

```python
from transformers import AutoModelForSequenceClassification

num_categories = len(le_cat.classes_)

model = AutoModelForSequenceClassification.from_pretrained(
    'allenai/scibert_scivocab_uncased',
    num_labels=num_categories
)
```

### 3.4 Training with Hugging Face Trainer

```python
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro'),
        'f1_weighted': f1_score(labels, preds, average='weighted')
    }

training_args = TrainingArguments(
    output_dir='./checkpoints',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1_macro',
    fp16=True,                          # Use if GPU supports it
    logging_steps=50,
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=PaperDataset(train_df['text'], train_df['cat_label']),
    eval_dataset=PaperDataset(val_df['text'], val_df['cat_label']),
    compute_metrics=compute_metrics
)

trainer.train()
```

### 3.5 Handling Class Imbalance

If some categories have far fewer samples, apply weighted loss:

```python
from torch import nn

class_weights = compute_class_weight('balanced', classes=np.unique(train_df['cat_label']), y=train_df['cat_label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = nn.CrossEntropyLoss(weight=class_weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss
```

---

## Phase 4 — Hierarchical Classification (Day 7–9)

Since you have both categories and subcategories, use a two-stage prediction approach.

### Strategy A — Sequential (simple, works well)

Train two separate classifiers: one for category, one for subcategory. At inference time, run the category model first, then run the subcategory model.

```
Paper text → [Category Model] → "Machine Learning"
                                       ↓
                             [Subcategory Model] → "Reinforcement Learning"
```

### Strategy B — Constrained Prediction (better accuracy)

At inference time, mask out subcategories that don't belong to the predicted category. This prevents nonsensical combinations like predicting category "Astronomy" with subcategory "Transformer Architecture."

```python
# Build a mapping: category → valid subcategory indices
cat_to_sub_indices = {}
for cat in le_cat.classes_:
    valid_subs = df[df['category'] == cat]['subcategory'].unique()
    cat_to_sub_indices[cat] = [le_sub.transform([s])[0] for s in valid_subs if s in le_sub.classes_]

def constrained_predict(text, cat_model, sub_model):
    predicted_cat = cat_model.predict([text])[0]
    cat_name = le_cat.inverse_transform([predicted_cat])[0]
    
    sub_logits = sub_model.predict_proba([text])[0]
    
    # Zero out logits for invalid subcategories
    mask = np.full(len(le_sub.classes_), -np.inf)
    for valid_idx in cat_to_sub_indices[cat_name]:
        mask[valid_idx] = sub_logits[valid_idx]
    
    predicted_sub = np.argmax(mask)
    return cat_name, le_sub.inverse_transform([predicted_sub])[0]
```

---

## Phase 5 — Evaluation (Day 9–10)

### 5.1 Metrics to Track

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Full classification report per class
print(classification_report(
    test_df['cat_label'],
    predictions,
    target_names=le_cat.classes_,
    digits=3
))

# Confusion matrix (use for top-N classes if too many)
cm = confusion_matrix(test_df['cat_label'], predictions)
sns.heatmap(cm, xticklabels=le_cat.classes_, yticklabels=le_cat.classes_, annot=True)
```

### 5.2 Error Analysis

Always look at what the model gets wrong — it reveals whether the issue is label noise, ambiguous papers, or a model weakness.

```python
errors = test_df.copy()
errors['predicted'] = le_cat.inverse_transform(predictions)
errors['actual'] = le_cat.inverse_transform(test_df['cat_label'])
errors = errors[errors['predicted'] != errors['actual']]

# Look at the most confused pairs
errors.groupby(['actual', 'predicted']).size().sort_values(ascending=False).head(20)
```

---

## Phase 6 — Serving the Model (Day 10–12)

### 6.1 Inference Pipeline

```python
class PaperClassifier:
    def __init__(self, model_path, tokenizer_path, cat_encoder_path, sub_encoder_path):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.cat_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.cat_model.eval()
        self.le_cat = joblib.load(cat_encoder_path)
    
    def predict(self, title: str, abstract: str, top_k: int = 3):
        text = f"{title} [SEP] {abstract}"
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            logits = self.cat_model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        top_indices = probs.argsort()[::-1][:top_k]
        
        return [
            {'category': self.le_cat.inverse_transform([i])[0], 'confidence': float(probs[i])}
            for i in top_indices
        ]
```

### 6.2 FastAPI Endpoint

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
classifier = PaperClassifier(...)

class PaperInput(BaseModel):
    title: str
    abstract: str

@app.post("/classify")
def classify_paper(paper: PaperInput):
    predictions = classifier.predict(paper.title, paper.abstract)
    return {"predictions": predictions}
```

---

## Phase 7 — Iteration and Improvement

Once you have a working baseline and a fine-tuned transformer, here are the most impactful ways to improve:

| Technique | Expected Gain | Effort |
|---|---|---|
| SciBERT → Longformer if abstracts are long | +2–4% F1 | Medium |
| Data augmentation (back-translation for rare classes) | +3–5% for tail classes | High |
| Ensemble TF-IDF + SciBERT (average probabilities) | +1–2% F1 | Low |
| Active learning: label ambiguous predictions manually | High long-term gain | High |
| Hyperparameter sweep (lr, batch size, warmup) | +1–3% F1 | Medium |

---

## Recommended Tech Stack

| Component | Tool |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Baseline ML | `scikit-learn` |
| Transformer models | `transformers` (Hugging Face), `torch` |
| Experiment tracking | `wandb` or `mlflow` |
| Serialization | `joblib`, `torch.save` |
| Serving | `fastapi` + `uvicorn` |
| Visualization | `matplotlib`, `seaborn` |

---

## Timeline Summary

| Phase | Task | Days |
|---|---|---|
| 0 | EDA + understand data | 1 |
| 1 | Data cleaning + label encoding | 1–2 |
| 2 | TF-IDF baseline | 2–3 |
| 3 | SciBERT fine-tuning | 3–7 |
| 4 | Hierarchical (category + subcategory) | 7–9 |
| 5 | Evaluation + error analysis | 9–10 |
| 6 | Serving (inference pipeline + API) | 10–12 |
| 7 | Iteration / improvement | 12+ |

---

## Quick Decision Guide

```
Do you have a GPU?
├── Yes → Go straight to SciBERT (Phase 3), skip baseline if pressed for time
└── No  → TF-IDF + LinearSVC baseline will get you 75–85% and runs on CPU in minutes
           Use Google Colab (free T4 GPU) for the transformer fine-tuning step

Are there >50 subcategories?
├── Yes → Use Strategy A (two separate models), it's simpler and more maintainable
└── No  → Strategy B (constrained prediction) is worth implementing for accuracy gains

Is the dataset heavily imbalanced?
├── Yes → Use weighted CrossEntropyLoss (Phase 3.5) + report per-class F1, not accuracy
└── No  → Standard training setup is fine
```

---

## Appendix — Install Commands

```bash
pip install transformers torch scikit-learn pandas numpy \
            fastapi uvicorn joblib matplotlib seaborn

# Optional but recommended
pip install wandb          # experiment tracking
pip install datasets       # HuggingFace dataset utilities
pip install sentencepiece  # required for some tokenizers
```
