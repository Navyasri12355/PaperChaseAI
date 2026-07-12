import json
import time
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelService:
    def __init__(self):
        self.loaded = False

    def load(self,
             main_model_path: str,
             sub_model_path: str,
             le_main_path: str,
             le_sub_path: str,
             hierarchy_path: str):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizers
        self.main_tokenizer = AutoTokenizer.from_pretrained(main_model_path)
        self.sub_tokenizer = AutoTokenizer.from_pretrained(sub_model_path)

        # Load models
        self.main_model = AutoModelForSequenceClassification.from_pretrained(main_model_path)
        self.main_model.to(self.device)
        self.main_model.eval()

        self.sub_model = AutoModelForSequenceClassification.from_pretrained(sub_model_path)
        self.sub_model.to(self.device)
        self.sub_model.eval()

        # Load label encoders and hierarchy
        self.le_main = joblib.load(le_main_path)
        self.le_sub = joblib.load(le_sub_path)
        self.hierarchy = json.load(open(hierarchy_path))

        self.loaded = True

    def predict(self, title: str, abstract: str, top_k: int = 3, min_confidence: float = 0.1) -> dict:
        start = time.time()
        text = f"{title} [SEP] {abstract}"

        # --- Main category inference ---
        main_inputs = self.main_tokenizer(
            text, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            main_logits = self.main_model(**main_inputs).logits

        main_probs = torch.softmax(main_logits, dim=-1).squeeze()
        
        # Get top k main categories
        top_main_probs, top_main_indices = torch.topk(main_probs, min(top_k, len(main_probs)))
        
        main_categories = []
        for prob, idx in zip(top_main_probs, top_main_indices):
            conf = prob.item()
            if conf >= min_confidence:
                label = self.le_main.inverse_transform([idx.item()])[0]
                main_categories.append({
                    "category": label,
                    "confidence": round(conf, 4)
                })

        # --- Sub category inference for each main category ---
        sub_inputs = self.sub_tokenizer(
            text, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sub_logits = self.sub_model(**sub_inputs).logits.squeeze()

        sub_categories = []
        
        # For each predicted main category, get top sub-categories
        for main_cat in main_categories:
            valid_subs = set(self.hierarchy.get(main_cat["category"], []))
            
            # Create a copy of logits for masking
            masked_logits = sub_logits.clone()
            for i, label in enumerate(self.le_sub.classes_):
                if label not in valid_subs:
                    masked_logits[i] = -1e9
            
            sub_probs = torch.softmax(masked_logits, dim=-1)
            top_sub_probs, top_sub_indices = torch.topk(sub_probs, min(top_k, len(sub_probs)))
            
            for prob, idx in zip(top_sub_probs, top_sub_indices):
                conf = prob.item()
                if conf >= min_confidence:
                    label = self.le_sub.inverse_transform([idx.item()])[0]
                    sub_categories.append({
                        "category": label,
                        "confidence": round(conf, 4)
                    })

        # Remove duplicate sub-categories and sort by confidence
        seen = set()
        unique_sub_categories = []
        for cat in sorted(sub_categories, key=lambda x: x["confidence"], reverse=True):
            if cat["category"] not in seen:
                seen.add(cat["category"])
                unique_sub_categories.append(cat)

        inference_time = (time.time() - start) * 1000

        return {
            "main_categories": main_categories,
            "sub_categories": unique_sub_categories[:top_k],
            "inference_time_ms": round(inference_time, 2)
        }

model_service = ModelService()
