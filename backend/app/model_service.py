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

    def predict(self, title: str, abstract: str) -> dict:
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
        main_idx = main_probs.argmax().item()
        main_conf = main_probs[main_idx].item()
        main_label = self.le_main.inverse_transform([main_idx])[0]

        # --- Sub category inference with constraint masking ---
        sub_inputs = self.sub_tokenizer(
            text, truncation=True, max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            sub_logits = self.sub_model(**sub_inputs).logits.squeeze()

        # Mask out sub-categories not belonging to predicted main category
        valid_subs = set(self.hierarchy.get(main_label, []))
        for i, label in enumerate(self.le_sub.classes_):
            if label not in valid_subs:
                sub_logits[i] = -1e9

        sub_probs = torch.softmax(sub_logits, dim=-1)
        sub_idx = sub_probs.argmax().item()
        sub_conf = sub_probs[sub_idx].item()
        sub_label = self.le_sub.inverse_transform([sub_idx])[0]

        inference_time = (time.time() - start) * 1000

        return {
            "main_category": main_label,
            "sub_category": sub_label,
            "main_confidence": round(main_conf, 4),
            "sub_confidence": round(sub_conf, 4),
            "inference_time_ms": round(inference_time, 2)
        }

model_service = ModelService()
