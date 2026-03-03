import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from tqdm.auto import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

MODEL_DIR = "model_run3"
MAX_LEN = 256
BATCH_SIZE = 64

# ================= LOAD TEST =================
test_df = pd.read_csv("test.csv")
TEXT_COL = "Text"

# ================= DATASET =================
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

# ================= LOAD MODEL =================
tokenizer = XLMRobertaTokenizer.from_pretrained(
    MODEL_DIR, local_files_only=True
)
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_DIR, local_files_only=True
).to(DEVICE)

model.eval()

loader = DataLoader(
    TestDataset(test_df[TEXT_COL].tolist(), tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# ================= PREDICT =================
pred_ids = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits

        pred_ids.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())

id2label = {0: "Non-Abusive", 1: "Abusive"}
pred_labels = [id2label[i] for i in pred_ids]

# ===== SAFE SUBMISSION FORMAT =====
if "ID" in test_df.columns:
    sub = pd.DataFrame({
        "ID": test_df["ID"],
        "Class": pred_labels,
    })
else:
    sub = pd.DataFrame({
        "Text": test_df[TEXT_COL],
        "Class": pred_labels,
    })

sub.to_csv("Infinity_Run3.csv", index=False, encoding="utf-8-sig")
print("✅ Saved Infinity_Run3.csv")
