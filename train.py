import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

# ================= SETTINGS =================
SEED = 42
MODEL_NAME = "xlm-roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-5
WEIGHT_DECAY = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)

# ================= SEED =================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ================= LOAD DATA =================
df = pd.read_csv("train.csv")

raw = df["Class"].astype(str)
norm = (
    raw.str.lower()
    .str.strip()
    .str.replace(" ", "", regex=False)
    .str.replace("-", "", regex=False)
)

label_map = {"nonabusive": 0, "abusive": 1}
df["label_id"] = norm.map(label_map)
df = df.dropna(subset=["label_id"]).copy()
df["label_id"] = df["label_id"].astype(int)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Text"].astype(str).tolist(),
    df["label_id"].tolist(),
    test_size=0.2,
    random_state=SEED,
    stratify=df["label_id"]
)

# ================= DATASET =================
class TamilDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ================= MODEL =================
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(DEVICE)

train_loader = DataLoader(
    TamilDataset(train_texts, train_labels, tokenizer),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    TamilDataset(val_texts, val_labels, tokenizer),
    batch_size=BATCH_SIZE,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

total_steps = len(train_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)

# ================= TRAIN LOOP =================
best_f1 = 0.0
os.makedirs("model_run3", exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - train"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    print(f"Train loss: {total_loss/len(train_loader):.4f}")

    # ===== VALIDATION =====
    model.eval()
    preds, gold = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            preds.extend(pred)
            gold.extend(batch["labels"].numpy().tolist())

    f1 = f1_score(gold, preds, average="macro")
    print("Validation Macro F1:", f1)

    if f1 > best_f1:
        best_f1 = f1
        model.save_pretrained("model")
        tokenizer.save_pretrained("model")
        print("Saved BEST model")

print("🔥 Best F1:", best_f1)
