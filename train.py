print("train.py started")

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


# -------- SETTINGS --------
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 128
BATCH_SIZE = 16          # smaller for CPU
EPOCHS = 5
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- LOAD DATA --------
df = pd.read_csv("data/train.csv")
print("Dataset loaded:", df.shape)

# IMPORTANT: check column names
print("Columns:", df.columns)

# Adjust column names if needed
TEXT_COL = "Text"
LABEL_COL = "Class"

# -------- ROBUST LABEL ENCODING (handles spaces, hyphens, casing) --------
raw = df[LABEL_COL].astype(str)

# normalize:
# 1) lowercase
# 2) strip leading/trailing spaces
# 3) remove all spaces
# 4) remove hyphens
norm = (
    raw.str.lower()
       .str.strip()
       .str.replace(" ", "", regex=False)
       .str.replace("-", "", regex=False)
)

print("Unique raw labels:", sorted(raw.unique()))
print("Unique normalized labels:", sorted(norm.unique()))

label_map = {
    "nonabusive": 0,
    "abusive": 1
}

df["label_id"] = norm.map(label_map)

# Drop any rows with unmapped/unknown labels (prevents NaN crash)
before = len(df)
df = df.dropna(subset=["label_id"]).copy()
df["label_id"] = df["label_id"].astype(int)

classes = np.array([0, 1])
weights = compute_class_weight(class_weight="balanced", classes=classes, y=df["label_id"].values)
class_weights = torch.tensor(weights, dtype=torch.float).to(DEVICE)
print("Class weights:", class_weights)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

after = len(df)

print(f"Label mapping done. Dropped {before-after} bad rows. Remaining: {after}")

# -------- TRAIN-VALIDATION SPLIT --------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df[TEXT_COL].astype(str).tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label_id"].tolist()
)

# -------- DATASET CLASS --------
class TamilDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# -------- TOKENIZER & MODEL --------
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.to(DEVICE)

train_dataset = TamilDataset(train_texts, train_labels, tokenizer)
val_dataset = TamilDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
best_f1 = 0.0

# -------- TRAINING LOOP --------
for epoch in range(EPOCHS):
    print(f"\n Epoch {epoch+1}/{EPOCHS}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
        input_ids=input_ids,
         attention_mask=attention_mask
        )

        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")

    # -------- VALIDATION --------
    model.eval()
    preds, true = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)
            preds.extend(predictions.cpu().numpy())
            true.extend(batch["labels"].numpy())

    f1 = f1_score(true, preds, average="macro")
    print(f"Validation Macro F1: {f1:.4f}")

# -------- SAVE MODEL --------
if f1 > best_f1:
    best_f1 = f1
    model.save_pretrained("model_best")
    tokenizer.save_pretrained("model_best")
    print(f"Saved BEST model (Macro F1={best_f1:.4f}) to ./model_best/")