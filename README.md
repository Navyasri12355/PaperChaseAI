# ArXiv Paper Hierarchical Classification — NLP Lab EL

A two-stage **SciBERT-based hierarchical classification system** that categorises ArXiv research papers into 8 main categories and 117 sub-categories using a *Constrained Sequential* strategy.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Web Application](#web-application)
- [Architecture & Methodology](#architecture--methodology)
- [Directory Structure](#directory-structure)
- [File Descriptions](#file-descriptions)
  - [Tracked Files (in this repository)](#tracked-files-in-this-repository)
  - [Excluded Files (not in this repository)](#excluded-files-not-in-this-repository)
- [Model Performance](#model-performance)
- [Hierarchy Map](#hierarchy-map)
- [How to Run the Application](#how-to-run-the-application)
- [How to Reproduce](#how-to-reproduce)
- [Dependencies](#dependencies)

---

## Project Overview

This project tackles **multi-class hierarchical text classification** on the ArXiv dataset. Given the title and abstract of a research paper, the system predicts:

1. **Main Category** — one of 8 broad fields (Computer Science, Physics, Mathematics, etc.)
2. **Sub-Category** — one of 117 fine-grained sub-fields (e.g., `Computer Science → Machine Learning`)

The pipeline uses **SciBERT** (`allenai/scibert_scivocab_uncased`), a BERT-based model pre-trained on scientific text, fine-tuned separately for each level of the hierarchy.

---

## Web Application

This project includes a **full-stack web application** that provides an interactive interface for classifying ArXiv papers in real-time.

### Tech Stack

- **Backend:** FastAPI (Python) — RESTful API serving the fine-tuned SciBERT models
- **Frontend:** React + TypeScript + Vite — Modern, responsive UI with Tailwind CSS
- **ML Inference:** PyTorch + Transformers — Real-time hierarchical classification with constraint masking

### Features

✅ **Real-time Classification** — Submit paper title and abstract, get instant predictions  
✅ **Hierarchical Constraint Enforcement** — Guarantees valid parent-child category pairs  
✅ **Confidence Scores** — Visual progress bars showing model confidence for both levels  
✅ **Inference Time Tracking** — Displays prediction latency in milliseconds  
✅ **Input Validation** — Client and server-side validation with helpful error messages  
✅ **Responsive Design** — Clean, modern UI with college branding

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns model loading status |
| `/categories` | GET | Returns the complete hierarchy map (8 main → 117 sub) |
| `/predict` | POST | Classifies a paper given `title` and `abstract` |

### Application Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (React)                        │
│  • Paper submission form with validation                    │
│  • Results display with confidence visualization            │
│  • Responsive UI with Tailwind CSS                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP POST /predict
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Backend (FastAPI)                         │
│  • Request validation (Pydantic schemas)                    │
│  • Model service singleton                                  │
│  • CORS middleware for local development                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Model Service Layer                        │
│  1. Load both SciBERT models on startup                     │
│  2. Tokenize input (title + [SEP] + abstract)              │
│  3. Predict main category                                   │
│  4. Predict sub-category with constraint masking           │
│  5. Return predictions + confidence + timing                │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture & Methodology

### Strategy: Constrained Sequential Classification

```
Input Paper (Title + Abstract)
         │
         ▼
 ┌───────────────────┐
 │  Main Category    │  ──► SciBERT fine-tuned on 8 classes
 │     Model         │       Accuracy: 84.08% | Macro F1: 77.26%
 └───────────────────┘
         │
         │  Predicted Main Category
         ▼
 ┌───────────────────┐
 │  Sub-Category     │  ──► SciBERT fine-tuned on 117 classes
 │     Model         │
 └───────────────────┘
         │
         │  Raw logits over all 117 sub-categories
         ▼
 ┌───────────────────────────────────────┐
 │  Constrained Masking                  │
 │  Zero-out logits for sub-categories   │
 │  NOT belonging to predicted parent    │
 └───────────────────────────────────────┘
         │
         ▼
  Final Sub-Category Prediction
  (Guaranteed to belong to predicted main category)
  Hierarchical Consistency: 100%
```

### Why Constrained Sequential?

| Property | Unconstrained | Constrained Sequential |
|---|---|---|
| Hierarchical consistency | 88.28% | **100%** |
| Joint accuracy | 46.51% | **49.29%** |
| Implementation complexity | Low | Low |
| Guarantees valid parent-child pair | ❌ | ✅ |

**Trade-offs acknowledged:**
- An error in the main category prediction propagates to the sub-category (error chaining)
- Two full forward passes are needed at inference time
- Each model can be independently updated or swapped without retraining the other

---

## Directory Structure

```
Lab EL/
│
├── backend/                        # FastAPI backend application
│   ├── app/
│   │   ├── main.py                 # FastAPI app with CORS and lifespan
│   │   ├── model_service.py        # Model loading and inference logic
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── routers/
│   │       ├── predict.py          # POST /predict endpoint
│   │       ├── health.py           # GET /health endpoint
│   │       └── categories.py       # GET /categories endpoint
│   ├── requirements.txt            # Python dependencies
│   └── .env                        # Environment configuration
│
├── frontend/                       # React + TypeScript frontend
│   ├── src/
│   │   ├── App.tsx                 # Main app component with layout
│   │   ├── main.tsx                # Vite entry point
│   │   ├── index.css               # Tailwind CSS imports
│   │   ├── api/
│   │   │   └── client.ts           # Axios API client
│   │   └── components/
│   │       ├── PaperForm.tsx       # Input form with validation
│   │       └── ResultsPanel.tsx    # Results display component
│   ├── public/
│   │   └── Emblem.png              # College emblem
│   ├── package.json                # Node dependencies
│   ├── vite.config.ts              # Vite config with proxy
│   ├── tailwind.config.js          # Tailwind CSS config
│   └── tsconfig.json               # TypeScript config
│
├── Main Category/                  # Fine-tuned SciBERT for main category
│   ├── config.json                 ✅ tracked
│   ├── tokenizer.json              ✅ tracked
│   ├── tokenizer_config.json       ✅ tracked
│   └── model.safetensors           ❌ NOT tracked (440 MB — see below)
│
├── Sub Category/                   # Fine-tuned SciBERT for sub-category
│   ├── training_log.csv            ✅ tracked
│   ├── training_curves.png         ✅ tracked
│   └── best_model/
│       ├── config.json             ✅ tracked
│       ├── tokenizer.json          ✅ tracked
│       ├── tokenizer_config.json   ✅ tracked
│       └── model.safetensors       ❌ NOT tracked (440 MB — see below)
│
├── baseline_results.csv            ✅ tracked
├── le_main_category.pkl            ✅ tracked
├── le_sub_category.pkl             ✅ tracked
├── phase4_confusion_joint.png      ✅ tracked
├── phase4_hierarchy_map.json       ✅ tracked
├── phase4_report.txt               ✅ tracked
├── phase4_results.csv              ✅ tracked
├── phase5_error_analysis.ipynb     ✅ tracked
├── implementation.md               ✅ tracked (full implementation plan)
├── train.csv                       ❌ NOT tracked (38 MB — see below)
├── test.csv                        ❌ NOT tracked (4.7 MB — see below)
├── val.csv                         ❌ NOT tracked (4.7 MB — see below)
├── .gitignore                      ✅ tracked
└── README.md                       ✅ tracked
```

---

## File Descriptions

### Tracked Files (in this repository)

#### `Main Category/`

| File | Description |
|---|---|
| `config.json` | HuggingFace model configuration for the main-category SciBERT classifier. Defines the BERT architecture: 12 hidden layers, 12 attention heads, hidden size 768, vocab size 31090 (SciBERT scientific vocabulary), 8 output labels (one per main category). |
| `tokenizer.json` | The full tokenizer definition in HuggingFace fast-tokenizer format. Contains the WordPiece vocabulary, special tokens (`[CLS]`, `[SEP]`, `[PAD]`, `[MASK]`, `[UNK]`), and tokenization rules. Required to preprocess input text identically to how the model was trained. |
| `tokenizer_config.json` | Lightweight tokenizer metadata: tokenizer class (`BertTokenizer`), case sensitivity (`do_lower_case: false`), special token identities, and max model length. |

#### `Sub Category/`

| File | Description |
|---|---|
| `training_log.csv` | Epoch-by-epoch training metrics for the sub-category model across 5 epochs: train loss, validation loss, validation macro F1, and validation accuracy. Shows steady improvement from F1=0.334 (epoch 1) to F1=0.472 (epoch 5). |
| `training_curves.png` | Visual plot of the training and validation loss/F1 curves across epochs. Useful for inspecting convergence and potential overfitting. |
| `best_model/config.json` | HuggingFace config for the sub-category SciBERT classifier. Same BERT architecture as the main model but with **117 output labels** — one per ArXiv sub-category. |
| `best_model/tokenizer.json` | Full tokenizer definition for the sub-category model (identical vocabulary and rules to the main category tokenizer, as both are SciBERT-based). |
| `best_model/tokenizer_config.json` | Tokenizer metadata for the sub-category model. |

#### Root-level files

| File | Description |
|---|---|
| `baseline_results.csv` | Summary CSV of the final evaluation metrics for all three evaluation modes: main category classification, unconstrained sub-category, and constrained sub-category. |
| `phase4_results.csv` | Detailed results table from Phase 4 (Hierarchical Classification), including joint accuracy and hierarchical consistency scores for each strategy. |
| `phase4_report.txt` | Full text classification report from Phase 4. Includes per-category joint accuracy breakdown across 8 main categories, and a per-sub-category precision/recall/F1 report across all 117 sub-categories on the test set. |
| `phase4_hierarchy_map.json` | JSON dictionary mapping each of the 8 main categories to its valid set of sub-category labels. This is used at inference time to construct the constraint mask, ensuring only valid parent-child predictions are output. |
| `phase4_confusion_joint.png` | Joint confusion matrix visualisation showing where the constrained hierarchical system makes errors — both correct and incorrect predictions across main and sub-category axes. |
| `le_main_category.pkl` | Scikit-learn `LabelEncoder` fitted on the 8 main category class names. Used to convert between integer model outputs and human-readable category strings (e.g., `0 → "Computer Science"`). |
| `le_sub_category.pkl` | Scikit-learn `LabelEncoder` fitted on the 117 sub-category labels. Used to decode sub-category model outputs to ArXiv sub-category strings (e.g., `42 → "LG"`). |
| `phase5_error_analysis.ipynb` | Jupyter notebook performing Phase 5 error analysis — inspecting misclassified papers, analysing failure modes across categories, and identifying patterns in model errors. |

---

### Excluded Files (not in this repository)

These files are listed in `.gitignore` and are **not committed** to this repository. They are excluded for the reasons detailed below.

---

#### `Main Category/model.safetensors` — **440 MB**

> **What it contains:** The complete set of fine-tuned neural network weights for the main-category SciBERT classifier. This is a `BertForSequenceClassification` model with ~110 million parameters stored in the `.safetensors` format (a safe, efficient binary tensor format by HuggingFace). The weights encode everything the model learned during fine-tuning on ArXiv main-category data.
>
> **Why excluded:** At **440 MB**, this file far exceeds GitHub's hard per-file limit of **100 MB**. Attempting to push it causes an HTTP 408 timeout error (as experienced). Even with Git LFS, the free tier only provides 1 GB of storage and bandwidth — not practical for two 440 MB model files. The model can be reproduced by fine-tuning SciBERT using the provided configs and training data, or loaded from a model hosting service (e.g., HuggingFace Hub).

---

#### `Sub Category/best_model/model.safetensors` — **440 MB**

> **What it contains:** The fine-tuned weights for the sub-category SciBERT classifier. Same architecture as the main model but with a **117-class classification head** (one per ArXiv sub-category). These weights were saved at the best validation checkpoint after 5 training epochs, achieving a validation macro F1 of 0.472.
>
> **Why excluded:** Same reason as above — **440 MB**, exceeding GitHub's 100 MB per-file limit. This is the "best_model" checkpoint that gives the performance numbers reported in `phase4_report.txt`. To use the model, it must be downloaded separately or retrained from scratch using the training script.

---

#### `train.csv` — **38 MB**

> **What it contains:** The training split of the ArXiv dataset. Each row represents one paper with columns for title, abstract, main category label, and sub-category label. This is the primary data used to fine-tune both SciBERT models.
>
> **Why excluded:** At **38 MB**, this file is at the upper boundary of what GitHub recommends (files over 50 MB trigger warnings; over 100 MB are rejected). More importantly, this is derived from the publicly available **ArXiv dataset** on Kaggle/Cornell, so it adds unnecessary repository bloat without adding original intellectual value. Collaborators should download the raw dataset and run the preprocessing pipeline to regenerate the splits.

---

#### `test.csv` — **4.7 MB**

> **What it contains:** The held-out test split of the ArXiv dataset (~1,903 papers). Used exclusively for final evaluation reported in `phase4_report.txt` and `phase4_results.csv`. Never seen by the model during training or validation.
>
> **Why excluded:** Derived from the same public ArXiv dataset. The evaluation results and metrics computed from this file are already fully documented in the committed result files (`phase4_report.txt`, `phase4_results.csv`, `baseline_results.csv`), so the raw CSV provides no additional reproducibility value in the repository itself.

---

#### `val.csv` — **4.7 MB**

> **What it contains:** The validation split (~same size as test). Used during training to monitor generalisation, select the best model checkpoint (highest val macro F1), and produce the `training_log.csv` metrics.
>
> **Why excluded:** Same reasoning as `test.csv` — derived from the public ArXiv dataset, and the validation metrics are already captured in `Sub Category/training_log.csv`.

---

## Model Performance

### Main Category Classification (8 classes)

| Metric | Score |
|---|---|
| Accuracy | **84.08%** |
| Macro F1 | **77.26%** |
| Weighted F1 | **84.56%** |

### Sub-Category Classification (117 classes)

| Strategy | Accuracy | Macro F1 | Joint Accuracy | Hierarchical Consistency |
|---|---|---|---|---|
| Unconstrained | 49.97% | 45.55% | 46.51% | 88.28% |
| **Constrained Sequential** | **49.29%** | **44.48%** | **49.29%** | **100.00%** |

### Per-Category Joint Accuracy (Constrained)

| Main Category | Test Samples | Cat Accuracy | Joint Accuracy |
|---|---|---|---|
| Quantitative Biology | 105 | 92.38% | 68.57% |
| Electrical Engineering | 68 | 64.71% | 64.71% |
| Economics | 52 | 76.92% | 59.62% |
| Physics | 548 | 89.05% | 54.56% |
| Quantitative Finance | 96 | 85.42% | 46.88% |
| Computer Science | 530 | 80.19% | 46.04% |
| Statistics | 77 | 79.22% | 41.56% |
| Mathematics | 427 | 85.01% | 40.05% |

---

## Hierarchy Map

The system supports **8 main categories** and **117 sub-categories**:

| Main Category | Number of Sub-Categories | Example Sub-Categories |
|---|---|---|
| Computer Science | 35 | AI, LG, CV, RO, SE, OS, DB |
| Physics | 31 | flu-dyn, quant-gas, chem-ph, plasm-ph, optics |
| Mathematics | 28 | AG, AT, DG, FA, HO, NT, RA |
| Quantitative Finance | 6 | CP, MF, PM, RM, PR, ST |
| Quantitative Biology | 6 | BM, MN, NC, PE, QM, GN |
| Statistics | 5 | ME, ML, OT, AP, CO |
| Electrical Engineering | 4 | AS, SP, IV, SY |
| Economics | 3 | EM, GN, TH |

The complete mapping is in [`phase4_hierarchy_map.json`](./phase4_hierarchy_map.json).

---

## How to Run the Application

### Prerequisites

- Python 3.8+ with pip
- Node.js 16+ with npm
- The two model files (`model.safetensors`) must be placed in their respective directories:
  - `Main Category/model.safetensors`
  - `Sub Category/best_model/model.safetensors`

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --reload --port 8000
```

The backend will:
- Load both SciBERT models on startup (may take 10-30 seconds)
- Start serving on `http://localhost:8000`
- Enable CORS for frontend at `http://localhost:5173`

### Frontend Setup

```bash
# Navigate to frontend directory (in a new terminal)
cd frontend

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

The frontend will:
- Start on `http://localhost:5173`
- Proxy API requests to the backend via `/api` prefix
- Hot-reload on file changes

### Using the Application

1. Open `http://localhost:5173` in your browser
2. Enter a paper **title** (3-300 characters)
3. Enter a paper **abstract** (50-3000 characters)
4. Click **"Classify Paper"**
5. View the predicted main category, sub-category, and confidence scores
6. Click **"Classify another paper"** to reset

### Example Input

**Title:**  
`Attention Is All You Need`

**Abstract:**  
`The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.`

**Expected Output:**  
- Main Category: `Computer Science` (high confidence)
- Sub-Category: `LG` (Machine Learning) or `CL` (Computation and Language)

---

## How to Reproduce

### 1. Prepare the dataset

Download the ArXiv dataset from [Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) and run the preprocessing script to generate `train.csv`, `val.csv`, and `test.csv`.

### 2. Fine-tune the Main Category model

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "allenai/scibert_scivocab_uncased", num_labels=8
)
# Fine-tune on train.csv, evaluate on val.csv
# Save to Main Category/
```

### 3. Fine-tune the Sub-Category model

Same as above but with `num_labels=117`. Save the best checkpoint to `Sub Category/best_model/`.

### 4. Run constrained inference

```python
import json, torch
import numpy as np

hierarchy = json.load(open("phase4_hierarchy_map.json"))
label_encoder_main = ...   # load le_main_category.pkl
label_encoder_sub  = ...   # load le_sub_category.pkl

# Step 1: Get main category prediction
main_logits = main_model(**inputs).logits
predicted_main = label_encoder_main.inverse_transform([main_logits.argmax()])[0]

# Step 2: Get sub-category logits
sub_logits = sub_model(**inputs).logits.squeeze()

# Step 3: Mask out invalid sub-categories
valid_subs = set(hierarchy[predicted_main])
for i, label in enumerate(label_encoder_sub.classes_):
    if label not in valid_subs:
        sub_logits[i] = -1e9  # zero-out invalid logits

predicted_sub = label_encoder_sub.inverse_transform([sub_logits.argmax()])[0]
```

### 5. Analyse errors

Open and run `phase5_error_analysis.ipynb` to inspect misclassifications on the test set.

---

## Dependencies

### Backend (Python)

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.2.0
joblib>=1.3.0
python-dotenv>=1.0.0
pydantic>=2.0.0
numpy>=1.24.0
safetensors>=0.3.0
```

Install via:
```bash
cd backend
pip install -r requirements.txt
```

### Frontend (Node.js)

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.0",
    "@tanstack/react-query": "^5.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "vite": "^5.0.0",
    "typescript": "^5.3.0",
    "tailwindcss": "^3.3.0",
    "@tailwindcss/forms": "^0.5.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
```

Install via:
```bash
cd frontend
npm install
```

### Model Training (for reproduction)

```
transformers>=4.30.0
torch>=2.0.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
jupyter>=1.0.0
safetensors>=0.3.0
```

Install via:
```bash
pip install transformers torch scikit-learn pandas numpy matplotlib jupyter safetensors
```

---

*This project was developed as part of the Natural Language Processing Lab — VIth Semester, RVCE.*

**Developed by:** Navyasri Mahitha Pulipati, Shreya Mohan, and Shravyaa S  
**Department:** Artificial Intelligence and Machine Learning  
**Institution:** RV College of Engineering
