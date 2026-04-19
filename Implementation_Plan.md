# NLP Research Paper Classifier — Implementation Plan

## Project Summary

Build a classifier that takes a research paper's title and abstract and predicts its category and subcategory. Input data: ~23,000 labeled papers in CSV format.

---

## Phase 0 — Data Audit

**Goal:** Know exactly what you're working with before making any decisions.

**Milestones:**
- [x] Count unique categories and subcategories
- [x] Plot class distribution for both — identify heavily imbalanced classes
- [x] Determine whether papers can have multiple categories (multi-label) or exactly one (multi-class)
- [x] Check for null/missing values in title, abstract, category, subcategory
- [x] Check for duplicate papers (same title or same abstract)
- [x] Measure abstract length distribution (min, median, max, % over 512 tokens)
- [x] Identify papers with title-only (no abstract) and decide how to handle them
- [x] Confirm the category–subcategory relationship (is every subcategory unique to one parent category, or do subcategories overlap across categories?)
- [x] Document findings in a short data card (one paragraph per finding)

**Exit Criteria:** You can answer these three questions with hard numbers — how many classes, how skewed is the distribution, and how much data is missing.

---

## Phase 1 — Data Preparation

**Goal:** Produce clean, encoded, split datasets ready for any model to consume.

**Milestones:**
- [x] Define a text cleaning strategy — decide what to strip (punctuation, LaTeX symbols, URLs, special characters) and what to keep (hyphens in compound terms, numbers)
- [x] Combine title and abstract into a single input field with a clear separator
- [x] Handle missing abstracts — decide: drop the row, use title only, or flag with a placeholder
- [x] Encode category and subcategory labels to integers; save the encoders to disk
- [x] Perform a stratified train/val/test split (suggested 80/10/10) stratified on category
- [x] Verify split: check that every category appears in all three splits
- [x] Verify split: check that class distribution roughly matches across splits
- [x] Save the three splits as separate CSV files for reproducibility

**Exit Criteria:** Three clean CSV files exist. You can load any split, run a value_counts() on labels, and confirm no label appears in val/test but not in train.

---

## Phase 2 — Baseline Model

**Goal:** Get a working end-to-end pipeline and a number to beat. No deep learning yet.

**Milestones:**
- [x] Build a TF-IDF + Logistic Regression pipeline for category prediction
- [x] Evaluate on val set — record accuracy, macro F1, and weighted F1
- [x] Build a TF-IDF + LinearSVC pipeline and compare against logistic regression
- [x] Report per-class F1 scores — identify the top 5 best and worst predicted classes
- [x] Run the same two pipelines for subcategory prediction
- [x] Log all results in a comparison table (model, task, accuracy, macro F1, runtime)
- [x] Identify whether errors cluster around specific class pairs (most confused categories)

**Exit Criteria:** A comparison table exists with documented macro F1 baselines for both category and subcategory prediction. All future models must beat these numbers.

---

## Phase 3 — Transformer Model

**Goal:** Fine-tune a pretrained language model that meaningfully outperforms the baseline.

**Milestones:**
- [x] Select a pretrained model — evaluate these options and pick one with justification:
  - `allenai/scibert_scivocab_uncased` — trained on scientific papers, best default choice
  - `microsoft/BiomedNLP-PubMedBERT-base-uncased` — if papers skew biomedical
  - `bert-base-uncased` — general fallback
  - `distilbert-base-uncased` — if GPU memory or inference speed is a hard constraint
- [x] Decide on max token length based on Phase 0 findings (512 for most BERT variants; consider Longformer if >30% of abstracts exceed 512 tokens)
- [x] Define the tokenization strategy — confirm title + abstract fits within max length; decide truncation behavior if it doesn't
- [x] Set up a training configuration: learning rate, batch size, number of epochs, warmup schedule, weight decay — document the reasoning for each choice
- [x] Decide how to handle class imbalance: weighted loss function, oversampling rare classes, or none
- [x] Train the category classifier; log train loss and val F1 per epoch
- [x] Confirm the model is learning (val F1 improves over epochs) — if not, diagnose before continuing
- [x] Apply early stopping based on val macro F1
- [x] Save the best checkpoint
- [x] Evaluate on the held-out test set — record accuracy, macro F1, weighted F1, and per-class F1
- [x] Compare results against the Phase 2 baseline in the comparison table

**Exit Criteria:** A saved model checkpoint exists. Test set macro F1 beats the TF-IDF baseline. Per-class F1 has been reviewed and no class sits at zero.

---

## Phase 4 — Hierarchical Classification (Category + Subcategory)

**Goal:** Extend the system to predict both category and subcategory in a coherent way.

**Milestones:**
- [ ] Map out the full category → subcategory hierarchy and confirm which subcategories belong to which parent
- [ ] Choose a hierarchical strategy and document the tradeoff:
  - **Sequential:** two separate models; at inference run category first, then subcategory
  - **Constrained:** same as sequential but mask out invalid subcategories based on the predicted parent
  - **Multi-output:** one model with two output heads sharing a backbone
- [ ] Train the subcategory classifier (same architecture as Phase 3)
- [ ] Evaluate subcategory prediction in isolation on the test set
- [ ] Implement the joint prediction logic — given a paper, produce both a category and subcategory
- [ ] Evaluate joint accuracy — how often are both category AND subcategory correct simultaneously
- [ ] Evaluate hierarchical consistency — how often does the predicted subcategory actually belong to the predicted category
- [ ] Log all metrics in the comparison table

**Exit Criteria:** The system returns a predicted category AND subcategory for any input. Joint accuracy and hierarchical consistency metrics are both recorded.

---

## Phase 5 — Error Analysis and Model Iteration

**Goal:** Understand failure modes and make targeted improvements before shipping.

**Milestones:**
- [ ] Extract all misclassified papers from the test set
- [ ] Identify the top 10 most confused category pairs; manually examine 5–10 examples per pair
- [ ] Hypothesize the root cause for each confused pair: label noise, topic overlap, short abstracts, domain jargon, or model limitation
- [ ] Decide on at least one targeted fix per root cause — e.g., label noise → audit and relabel; rare class → oversample; topic overlap → consider merging
- [ ] Implement the fix, retrain, and confirm improvement on the previously confused pairs
- [ ] Re-evaluate on the full test set — confirm overall metrics did not regress
- [ ] Document what was tried, what worked, and what didn't

**Exit Criteria:** At least one targeted fix has been implemented and validated. The confusion matrix has visibly improved on the previously worst pairs.

---

## Phase 6 — Inference Pipeline

**Goal:** Wrap the model in a clean, reusable interface that can be called from anywhere.

**Milestones:**
- [ ] Define the input/output contract: input is title + abstract (strings); output is top-K predictions each with category, subcategory, and confidence score
- [ ] Build a self-contained inference class that loads model, tokenizer, and label encoders from disk
- [ ] Confirm inference is correct on 10 manually chosen test papers
- [ ] Measure latency on a single sample and on a batch of 100
- [ ] Confirm the pipeline handles edge cases: empty abstract, very long abstract, unknown characters
- [ ] Build a simple REST API with a single POST /classify endpoint
- [ ] Test the API end-to-end with a sample request
- [ ] Document the API: input schema, output schema, example request/response

**Exit Criteria:** A running API accepts a paper and returns a structured prediction. Latency is measured and acceptable for your use case.

---

## Phase 7 — Monitoring and Roadmap

**Goal:** Make the system maintainable and set up a path for continuous improvement.

**Milestones:**
- [ ] Define what model drift looks like here — what signals indicate the model is degrading?
- [ ] Set up prediction logging: record input hash, predicted label, and confidence for every inference
- [ ] Define a retraining trigger — e.g., 500 new labeled papers, or val F1 drops below a threshold
- [ ] Write a retraining runbook so anyone on the team can reproduce the full pipeline
- [ ] Prioritize and document next improvement experiments:
  - Longformer for papers with long abstracts
  - Ensemble of TF-IDF + transformer (averaged probabilities)
  - Data augmentation for rare subcategories via back-translation
  - Active learning: flag low-confidence predictions for human review

**Exit Criteria:** A retraining runbook exists. At least one future experiment is documented with a hypothesis and expected outcome.

---

## Milestone Tracker

| Phase | Description | Status |
|---|---|---|
| 0 | Data Audit | ✅ |
| 1 | Data Preparation | ✅ |
| 2 | Baseline Model | ✅ |
| 3 | Transformer Model | — |
| 4 | Hierarchical Classification | — |
| 5 | Error Analysis + Iteration | — |
| 6 | Inference Pipeline | — |
| 7 | Monitoring + Roadmap | — |

---

## Key Decisions to Lock In Before Phase 3

These affect everything downstream. Decide and document your reasoning before training anything.

1. **Multi-class vs multi-label** — can a paper have more than one category?
2. **Token length** — what percentage of your papers exceed 512 tokens?
3. **Base model** — SciBERT unless you have a strong reason not to
4. **Hierarchical strategy** — constrained sequential is the safest starting point
5. **Primary metric** — use macro F1, not accuracy, if classes are imbalanced
