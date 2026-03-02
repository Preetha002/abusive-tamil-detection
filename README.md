Infinity: Abusive Tamil Text Detection System

DravidianLangTech@ACL 2026 Shared Task

Overview
In this case, the code Infinity, a transformer based system to detect the abusive Tamil text against the women on social media has been stored.

The XLM-RoBERTa system is trained to binary classification ( Abusive / Non-Abusive ), and to work on the DravidianLangTech@ACL 2026 shared task.

Model
Base Model: xlm-roberta-base

Task: Binary Text Classification

Evaluation Metric: Macro F1 Score

Project Structure
train.py        # Training script
predict.py      # Prediction script
requirements.txt
README.md
.gitignore

Installation

Install dependencies:
pip install -r requirements.txt

Training
python train.py

Prediction
python predict.py

Note

Datasets and trained model weights are not included in this repository due to size constraints.