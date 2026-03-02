import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from tqdm import tqdm

MODEL_DIR = "model_run3"
MAX_LEN = 128
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
test_df = pd.read_csv("data/test.csv")

# Column name in your test file
TEXT_COL = "Text"

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
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze()
        }

# Load tokenizer + model
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
model.to(DEVICE)
model.eval()

dataset = TestDataset(test_df[TEXT_COL].tolist(), tokenizer)
loader = DataLoader(dataset, batch_size=BATCH_SIZE)

pred_ids = []

with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        pred_ids.extend(preds.cpu().numpy().tolist())

# Convert ids back to label names (match competition)
id2label = {0: "Non-Abusive", 1: "Abusive"}
pred_labels = [id2label[i] for i in pred_ids]

# Create submission file
submission = pd.DataFrame({
    "Text": test_df[TEXT_COL],
    "Class": pred_labels
})

submission.to_csv("Infinity_Run3.csv", index=False, encoding="utf-8-sig")
print("Saved: Infinity_Run3.csv")